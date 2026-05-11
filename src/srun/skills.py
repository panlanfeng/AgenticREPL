"""Skills system — loadable prompt templates from ~/.srun/skills/."""

import os
import glob


SKILLS_DIR = os.path.join(os.path.expanduser("~"), ".srun", "skills")


class Skill:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.content = ""
        self._load()

    def _load(self):
        try:
            with open(self.path) as f:
                self.content = f.read()
        except Exception:
            self.content = ""

    def prompt(self):
        """Return the skill as an injectable system message block."""
        return (
            f"[Active skill: {self.name}]\n"
            f"Apply the following instructions:\n\n"
            f"{self.content}\n\n"
            f"[/Active skill: {self.name}]"
        )


def load_skills():
    """Load all skills from SKILLS_DIR."""
    os.makedirs(SKILLS_DIR, exist_ok=True)
    skills = []
    for path in sorted(glob.glob(os.path.join(SKILLS_DIR, "*.md"))):
        name = os.path.splitext(os.path.basename(path))[0]
        skill = Skill(name, path)
        if skill.content.strip():
            skills.append(skill)
    return skills


def get_skill_prompts():
    """Return combined system message block for all loaded skills."""
    skills = load_skills()
    if not skills:
        return ""
    blocks = [s.prompt() for s in skills]
    return "\n\n".join(blocks)
