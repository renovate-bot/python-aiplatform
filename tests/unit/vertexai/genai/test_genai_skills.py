# //third_party/py/google/cloud/aiplatform/tests/unit/vertexai/genai/test_genai_skills.py
import json
from unittest import mock
from vertexai import _genai as genai
from vertexai._genai import client as vertexai_client
from google.genai import types as genai_types
import pytest


@pytest.fixture
def skills_client():
    creds = mock.MagicMock()
    creds.token = "test_token"
    client = vertexai_client.Client(
        project="test-project", location="test-location", credentials=creds
    )
    return client.skills


class TestGenaiSkills:
    mock_get_skill_response = {
        "name": "projects/test-project/locations/test-location/skills/test-skill",
        "displayName": "My Test Skill",
    }

    def test_get_skill(self, skills_client):
        """Tests the get_skill method."""
        with mock.patch.object(skills_client._api_client, "request") as request_mock:
            request_mock.return_value = genai_types.HttpResponse(
                body=json.dumps(self.mock_get_skill_response)
            )
            skill_name = (
                "projects/test-project/locations/test-location/skills/test-skill"
            )
            skill = skills_client.get(name=skill_name)
            request_mock.assert_called_with(
                "get",
                skill_name,
                {"_url": {"name": skill_name}},
                None,
            )
            assert isinstance(skill, genai.types.Skill)
            assert skill.name == skill_name
            assert skill.display_name == "My Test Skill"
