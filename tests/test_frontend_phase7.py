"""Tests for Frontend Phase 7: Azure Containerization & Packaging."""

from pathlib import Path
import tomllib


def test_dockerfile_exists_and_configured():
    """Verify Dockerfile exists and contains required production directives."""
    dockerfile_path = Path("Dockerfile")
    assert dockerfile_path.exists(), "Dockerfile must exist at repository root"

    content = dockerfile_path.read_text(encoding="utf-8")
    assert "FROM python:3.12-slim" in content
    assert "STREAMLIT_SERVER_PORT=8501" in content
    assert "HEALTHCHECK" in content
    assert "qcouser" in content, "Dockerfile should enforce non-root user execution"
    assert "EXPOSE 8501" in content


def test_dockerignore_exists():
    """Verify .dockerignore excludes virtualenv and sensitive cache files."""
    dockerignore_path = Path(".dockerignore")
    assert dockerignore_path.exists(), ".dockerignore must exist at repository root"

    content = dockerignore_path.read_text(encoding="utf-8")
    assert ".venv" in content
    assert "__pycache__" in content
    assert ".git" in content


def test_docker_compose_exists():
    """Verify docker-compose.yml defines standard container service."""
    compose_path = Path("docker-compose.yml")
    assert compose_path.exists(), "docker-compose.yml must exist at repository root"

    content = compose_path.read_text(encoding="utf-8")
    assert "qco-app:" in content
    assert "8501:8501" in content


def test_azure_deployment_guide_exists():
    """Verify azure_deployment_guide.md is present with deployment instructions."""
    guide_path = Path("deploy/azure_deployment_guide.md")
    assert guide_path.exists(), "deploy/azure_deployment_guide.md must exist"

    content = guide_path.read_text(encoding="utf-8")
    assert "Azure App Service" in content
    assert "Azure Container Apps" in content
    assert "az webapp" in content


def test_streamlit_config_production():
    """Verify .streamlit/config.toml is configured for containerized deployment."""
    config_path = Path(".streamlit/config.toml")
    assert config_path.exists()

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    assert data.get("server", {}).get("port") == 8501
    assert data.get("server", {}).get("headless") is True
    assert data.get("server", {}).get("address") == "0.0.0.0"
