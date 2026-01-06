import hashlib

def assign_variant(user_id: int, split: float = 0.5) -> str:
    digest = hashlib.md5(str(user_id).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 16**8
    return "A" if bucket < split else "B"

def assign_variants(users_df, split: float = 0.5):
    df = users_df.copy()
    df["variant"] = df["user_id"].apply(lambda x: assign_variant(int(x), split))
    return df