use anyhow::Result;
use tokenizers::Tokenizer;

use crate::models::chat::{ContentPart, Message, MessageRole};

///功能：将聊天消息编码为 Qwen 系列专用的 ChatML 提示词格式
/// 同时收集多模态占位符（图像在token序列中的位置）
///
///  设计极简：足够驱动模型运行，同时兼容 CLI 命令行 + 类 OpenAI API 请求
/// 定义公共结构体：PromptEncoder
// 核心作用：存储编码提示词需要的**所有特殊token ID**
pub struct PromptEncoder {
    pub image_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,

    pub im_start_token_id: u32,
    pub im_end_token_id: u32,
}


/// 定义公共结构体：ImageSpan
// 核心作用：标记一段token序列的起止位置 → 告诉模型“这里要放图像特征”
#[derive(Debug, Clone)]
pub struct ImageSpan {
    /// Start index in the token sequence (inclusive).
    pub start: usize,
    /// End index in the token sequence (exclusive).
    pub end: usize,
}
/// 为PromptEncoder 结构体实现方法
impl PromptEncoder {
    pub fn from_tokenizer(
        tokenizer: &Tokenizer,
        image_token_id: u32,
        vision_start_token_id: u32,
        vision_end_token_id: u32,
    ) -> Result<Self> {
        let im_start_token_id = tokenizer
            .token_to_id("<|im_start|>")
            .ok_or_else(|| anyhow!("tokenizer missing <|im_start|>"))?;
        let im_end_token_id = tokenizer
            .token_to_id("<|im_end|>")
            .ok_or_else(|| anyhow!("tokenizer missing <|im_end|>"))?;
        Ok(Self {
            image_token_id,
            vision_start_token_id,
            vision_end_token_id,
            im_start_token_id,
            im_end_token_id,
        })
    }
    /// 定义一个【私有静态函数】：把文本编码成 token ID 列表
    fn encode_text(tokenizer: &Tokenizer, s: &str) -> Result<Vec<u32>> {
        Ok(tokenizer
            .encode(s, false)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec())
    }
    /// 【实例方法】：推送聊天消息的【头部】
    // 作用：生成 <|im_start|>user\n  或 <|im_start|>assistant\n
    fn push_chat_header(&self, tokenizer: &Tokenizer, ids: &mut Vec<u32>, role: &str) -> Result<()> {
        ids.push(self.im_start_token_id);
        ids.extend(Self::encode_text(tokenizer, role)?);
        ids.extend(Self::encode_text(tokenizer, "\n")?);
        Ok(())
    }
    // 【实例方法】：推送聊天消息的【尾部】
    // 作用：生成 <|im_end|>\n
    fn push_chat_footer(&self, tokenizer: &Tokenizer, ids: &mut Vec<u32>) -> Result<()> {
        ids.push(self.im_end_token_id);
        ids.extend(Self::encode_text(tokenizer, "\n")?);
        Ok(())
    }

    // 定义公共实例方法：encode
    // 功能：把聊天消息 → (token_ids, 图片位置区间)
    pub fn encode(
        &self,
        tokenizer: &Tokenizer,
        messages: &[Message],
        image_token_counts: &[usize],
    ) -> Result<(Vec<u32>, Vec<ImageSpan>)> {
        let mut ids: Vec<u32> = vec![];
        let mut spans: Vec<ImageSpan> = vec![];
        let mut img_idx = 0usize;
        // 4. 遍历每一条聊天消息（系统、用户、助手）
        for m in messages {
            let role_str = match m.role {
                MessageRole::System => "system",
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
            };

            self.push_chat_header(tokenizer, &mut ids, role_str)?;

            match &m.content {
                // 情况 A：消息是纯文本
                crate::models::chat::MessageContent::Text(s) => {
                    ids.extend(Self::encode_text(tokenizer, s)?);
                }
                // 情况 B：消息是多个部分（文本、图片）
                crate::models::chat::MessageContent::Parts(parts) => {
                    for part in parts {
                        match part {
                            ContentPart::Text { text } => {
                                ids.extend(Self::encode_text(tokenizer, text)?);
                            }
                            ContentPart::ImageBase64 { .. } | ContentPart::ImageUrl { .. } => {
                                let n = *image_token_counts
                                    .get(img_idx)
                                    .ok_or_else(|| anyhow!("missing image_token_count for image #{img_idx}"))?;
                                img_idx += 1;

                                ids.push(self.vision_start_token_id);
                                let start = ids.len();
                                ids.extend(std::iter::repeat(self.image_token_id).take(n));
                                let end = ids.len();
                                ids.push(self.vision_end_token_id);
                                spans.push(ImageSpan { start, end });
                            }
                        }
                    }
                }
            }

            self.push_chat_footer(tokenizer, &mut ids)?;
        }

        // Add the start of an assistant message for the model to complete.
        self.push_chat_header(tokenizer, &mut ids, "assistant")?;

        Ok((ids, spans))
    }
}
