import tiktoken

# 2) Select the encoding for the model you intend to use.
encoding = tiktoken.encoding_for_model("gpt-4-turbo")

# 3) Read your Markdown file into memory:
with open("marshmallow.md", "r", encoding="utf-8") as f:
    markdown_text = f.read()

# 4) Tokenize the text:
tokens = encoding.encode(markdown_text)

# 5) Count the tokens:
print(f"Token count: {len(tokens)}")