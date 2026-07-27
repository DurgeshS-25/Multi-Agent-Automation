from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # API keys
    tavily_api_key: str
    gemini_api_key: str

    # Model + search config (defaults match .env.example)
    gemini_model: str = "gemini-2.0-flash"
    search_depth: str = "basic"
    max_search_results: int = 5

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


# Single shared instance imported everywhere
settings = Settings()