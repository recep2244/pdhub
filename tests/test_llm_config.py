from protein_design_hub.core.config import LLMConfig


def test_llm_config_resolve_ollama_defaults():
    cfg = LLMConfig(provider="ollama", base_url="", model="", api_key="")
    resolved = cfg.resolve()
    assert resolved.base_url == "http://localhost:11434/v1"
    assert resolved.model == "llama3.2:latest"
    assert resolved.api_key == "ollama"


def test_llm_config_resolve_env_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = LLMConfig(provider="openai", base_url="", model="", api_key="")
    resolved = cfg.resolve()
    assert resolved.base_url == "https://api.openai.com/v1"
    assert resolved.model == "gpt-4o"
    assert resolved.api_key == "sk-test"


def test_llm_config_resolve_custom_overrides():
    cfg = LLMConfig(
        provider="custom",
        base_url="http://localhost:9999/v1",
        model="custom-model",
        api_key="secret",
    )
    resolved = cfg.resolve()
    assert resolved.base_url == "http://localhost:9999/v1"
    assert resolved.model == "custom-model"
    assert resolved.api_key == "secret"
