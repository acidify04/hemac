Describe here all the security policies in place on this repository to help your contributors to handle security issues efficiently.

## Goods practices to follow

:warning:**You must never store credentials information into source code or config file in a GitHub repository**
- Block sensitive data being pushed to GitHub by git-secrets or its likes as a git pre-commit hook
- Audit for slipped secrets with dedicated tools
- Use environment variables for secrets in CI/CD (e.g. GitHub Secrets) and secret managers in production

# Security Policy

## Supported Versions

Use this section to tell people about which versions of your project are currently being supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |
| others  | :x:                |

Only the main branch is supported for academic/research use under the Thales Non-Production License Agreement. No production or commercial usage is permitted.
## Reporting a Vulnerability

You can ask for support by contacting oss@thalesgroup.com

## Disclosure policy

Please describe the security issue to oss@thalesgroup.com

## Security Update policy

Updates about security vulnerabilities found will be done here.

## Security related configuration

Settings users should consider that would impact the security posture of deploying this project, such as HTTPS, authorization and many others.

## Known security gaps & future enhancements

Security improvements we haven’t gotten to yet:
None
If you know how to implement one of the securitys control listed above, we welcome any contribution!
