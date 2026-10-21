"""Enable coverage in subprocesses launched by the e2e tests.

Without this, every CLI invocation is a fresh interpreter and the coverage report
claims cli.py/pipeline.py are 0% covered while the e2e suite exercises them on
every test — a number that looks like a gap but is an artefact.
"""
try:
    import coverage
    coverage.process_startup()
except Exception:
    pass
