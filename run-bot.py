import discord
from discord.ext import commands
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F
import os
from dotenv import load_dotenv

TOXICITY_THRESHOLD = 0.6
SPAN_THRESHOLD = 0.4
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
AVG_TOKENS = 90

# Load model
class ToxicityModel(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)

        # Heads
        self.seq_head = nn.Linear(hidden, 1)   # Sequence classification
        self.tok_head = nn.Linear(hidden, 2)   # Token-level classification

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        last_hidden = self.dropout(out.last_hidden_state)  # [B, T, H]
        cls_pooled = self.dropout(last_hidden[:, 0])      # CLS pooling

        seq_logits = self.seq_head(cls_pooled)            # [B, 1]
        tok_logits = self.tok_head(last_hidden)           # [B, T, 2]
        return seq_logits, tok_logits

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ToxicityModel(model_name="microsoft/deberta-v3-base")
checkpoint_path = "models/toxicity_span_epoch5.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)
model.to(device)

intents = discord.Intents.default()
intents.message_content = True  # Needed to read messages
bot = commands.Bot(command_prefix="!", intents=intents)

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

def predict_toxicity(text):
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=AVG_TOKENS,  # must match training
        return_offsets_mapping=True,
        return_tensors="pt"
    )
    offset_mapping = inputs.pop("offset_mapping")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run model
    with torch.no_grad():
        clf_logits, tok_logits = model(**inputs)

        # Sequence-level probability
        toxicity_prob = torch.sigmoid(clf_logits).squeeze(-1)

        # Token-level probabilities (take class=1 for toxic)
        token_probs = F.softmax(tok_logits, dim=-1)[..., 1][0]
        tok_preds = (token_probs >= SPAN_THRESHOLD).long()

    # Decide overall toxic/non-toxic
    is_toxic = toxicity_prob >= TOXICITY_THRESHOLD

    return is_toxic, tok_preds, offset_mapping

def censor_toxic_spans(text, tok_preds, offset_mapping):
    """
    Censor toxic spans by replacing each word with the appropriate number of stars
    """
    import re
    
    toks_to_censor = []
    for pred, (start, end) in zip(tok_preds, offset_mapping[0].tolist()):
        if pred == 1 and start != end:
            toks_to_censor.append((start, end))
    
    if not toks_to_censor:
        return None
    
    # Merge overlapping spans
    merged_spans = []
    for start, end in sorted(toks_to_censor):
        if not merged_spans or start > merged_spans[-1][1]:
            merged_spans.append([start, end])
        else:
            merged_spans[-1][1] = max(merged_spans[-1][1], end)
    
    # Build censored text by working backwards to avoid index shifts
    censored_text = text
    
    for start, end in reversed(merged_spans):
        # Get the original span (including surrounding context for spacing)
        original_span = text[start:end]
        
        if not original_span.strip():
            continue
        
        # Use regex to find and replace words while preserving spacing
        def replace_word(match):
            word = match.group()
            return '\\*' * len(word)  # Escape asterisk for Discord
        
        # Replace all word characters with asterisks, keeping spaces and punctuation
        replacement = re.sub(r'\w', lambda m: '\\*', original_span)
        
        # Ensure proper spacing around censored words
        # Check if we need to add space before
        if start > 0 and text[start-1].isalnum() and replacement.lstrip() == replacement:
            replacement = ' ' + replacement
        
        # Check if we need to add space after  
        if end < len(text) and text[end].isalnum() and replacement.rstrip() == replacement:
            replacement = replacement + ' '
        
        # Replace in the text
        censored_text = censored_text[:start] + replacement + censored_text[end:]
    
    return censored_text

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")

@bot.event
async def on_message(message):
    # Ignore the bots messages
    if message.author == bot.user:
        return

    is_toxic, tok_preds, offset_mapping = predict_toxicity(message.content)

    # We shall follow a conservative approach:
    # 1. If the message is not toxic -> do nothing
    # 2. if the message is toxic:
    #   1. If the spans list is not empty -> simply censor the toxic parts with '*'
    #   2. If the spans list is empty -> delete the message and inform the user via a dm
    if is_toxic:
        censored_text = censor_toxic_spans(message.content, tok_preds, offset_mapping)
        
        if censored_text:
            # Delete original and send censored
            try:
                await message.delete()
                await message.channel.send(f"{message.author.mention}: {censored_text}")
            except Exception as e:
                print(f"Error censoring message: {e}")
        else:
            try:
                await message.delete()
                await message.author.send(f"Message flagged due to toxicity:\n{message.content}")
            except Exception as e:
                print(f"Error handling toxic message: {e}")

    await bot.process_commands(message)

# Run your bot
bot.run(BOT_TOKEN)