# Security Policy
We use only public datasets (or synthetic). You should never commit PII/PHI.
If you find a vulnerability (in code, CI/CD, or dependencies):
- Create a Private Security Advisory on GitHub (Security → Advisories) or email nnenashevamipt@gmail.com.
- Do not create a public issue with exploitation details.
Scope: code in this repo, GitHub Actions, Dockerfile, demo API.

## Supported versions
main

## Handling secrets
We store secrets (any sensitive information) in GitHub Actions Secrets/Environments. 
Locally, we store them via .env (see .env.example).
