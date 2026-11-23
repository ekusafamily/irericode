from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class TestConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')
    openai_model: str = Field(default='gpt-4o-mini')

c = TestConfig()
print(f'Model: {c.openai_model}')