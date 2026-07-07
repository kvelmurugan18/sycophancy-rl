def get_hf_token():
    try:
        from dotenv import dotenv_values
    except ImportError:
        dotenv_values = None

    if dotenv_values is not None:
        try:
            values = dotenv_values()
        except Exception:
            values = None
        if values:
            token = values.get("HUGGINGFACE_TOKEN") or values.get("HF_TOKEN")
            if token:
                return token

    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        UserSecretsClient = None

    if UserSecretsClient is not None:
        try:
            secret = UserSecretsClient().get_secret("HUGGINGFACE_TOKEN")
        except Exception:
            secret = None
        if secret:
            return secret

    raise RuntimeError(
        "Hugging Face token not found. Set HUGGINGFACE_TOKEN (or HF_TOKEN) in a "
        ".env file accessible to python-dotenv, or add it as a Kaggle secret "
        "named HUGGINGFACE_TOKEN."
    )
