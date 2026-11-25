# Local container execution

Prerequisite: Docker with Linux containers enabled. Run from the repository root.

```sh
docker build -t local-project .
docker run --name local-project-run local-project
docker cp local-project-run:/app/output ./container-output
docker rm local-project-run
```

Batch synthetic-data pipeline. Copy output charts and CSVs from the stopped container.

The configuration is proposed and has not been built or runtime-tested. Dependency versions are not fully locked. Review the build context for secrets, private datasets and redistribution rights before building. Do not publish container images until licensing and security review are complete. If Phase 2 introduces a dependency manifest, consolidate the Docker install step with it after merge.
