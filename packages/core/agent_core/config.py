from dataclasses import dataclass, field
import os
from pathlib import Path

@dataclass
class Settings:
    database_url: str = field(default_factory=lambda: os.environ['DATABASE_URL'])
    litellm_url: str = field(default_factory=lambda: os.getenv('LITELLM_URL', 'http://localhost:4000/v1'))
    litellm_key: str = field(default_factory=lambda: os.environ['LITELLM_API_KEY'])
    api_key: str = field(default_factory=lambda: os.environ['AGENT_API_KEY'])
    text_model: str = field(default_factory=lambda: os.getenv('TEXT_MODEL', 'qwen3.6-35b-a3b-q4'))
    image_model: str = field(default_factory=lambda: os.getenv('IMAGE_MODEL', 'moss-image'))
    search_url: str = field(default_factory=lambda: os.getenv('SEARXNG_URL', 'http://localhost:8080'))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv('DATA_DIR', '/data')))
    skills_dir: Path = field(default_factory=lambda: Path(os.getenv('SKILLS_DIR', '/app/agents/news/skills')))
    public_url: str = field(default_factory=lambda: os.getenv('PUBLIC_URL', 'http://localhost:8000/agents/news'))
    a2a_url: str = field(default_factory=lambda: os.getenv('A2A_URL', 'http://localhost:8000/a2a'))
    trust_sso: bool = field(default_factory=lambda: os.getenv('TRUST_SSO', 'false').lower() == 'true')
