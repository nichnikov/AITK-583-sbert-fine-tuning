import os
from collections import namedtuple
from pydantic import (BaseModel, 
                      BaseSettings)
from src.config import PROJECT_ROOT_DIR


class TextsDeleteSample(BaseModel):
    """Схема данных для удаления данных по тексту из Индекса"""
    Index: str
    Texts: list[str]
    FieldName: str
    Score: float


ROW = namedtuple("ROW", "SysID, ID, Cluster, ParentModuleID, ParentID, ParentPubList, "
                        "ChildBlockModuleID, ChildBlockID, ModuleID, Topic, Subtopic, DocName, ShortAnswerText")


class Settings(BaseSettings):
    """Base settings object to inherit from."""

    class Config:
        env_file = os.path.join(PROJECT_ROOT_DIR, "data", ".env")
        env_file_encoding = "utf-8"


class ElasticSettings(Settings):
    """Elasticsearch settings."""

    hosts: str
    index: str
    user_name: str | None
    password: str | None

    max_hits: int = 100
    chunk_size: int = 100

    @property
    def basic_auth(self) -> tuple[str, str] | None:
        """Returns basic auth tuple if user and password are specified."""
        print(self.user_name, self.password)
        if self.user_name and self.password:
            return self.user_name, self.password
        return None
