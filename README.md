# oda-bot

should be configured per-site

Watches updates on github, and choose what to update in the dispatcher-chart. Once pushed to site's gitlab, CI/CD deploys it there.

This way control of deployment is in the site, but can be customly configured to be triggered by changes.

## Generate an ODA workflow notebook from a paper

Install the optional PDF support and configure an OpenAI-compatible API key:

```bash
pip install '.[paper-agent]'
export OPENAI_API_KEY=...
odabot generate-workflow-notebook 2301.00001 --output reproduction.ipynb
```

The paper source can also be a local PDF path or an arXiv paper URL. Set
`OPENAI_MODEL` and `OPENAI_BASE_URL` to select the model and API endpoint.
Generated notebooks are not executed or deployed; review their code and
scientific assumptions before running them. Use `--force` to replace an
existing output file.
