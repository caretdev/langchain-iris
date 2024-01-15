import pytest
from testcontainers.iris import IRISContainer


def pytest_addoption(parser):
    group = parser.getgroup("iris")

    group.addoption(
        "--embedded",
        action="store_true",
        help="Use embedded mode",
    )

    group.addoption(
        "--dburi",
        action="store",
        type=str,
        help="IRIS Connection Uri",
    )

    group.addoption(
        "--container",
        action="store",
        default=None,
        type=str,
        help="Docker image with IRIS",
    )


def pytest_configure(config: pytest.Config):
    global iris
    iris = None
    if config.option.embedded:
        config.option.dburi = "iris+emb:///"
        return
    if not config.option.container:
        return
    config.option.embedded = False
    print("Starting IRIS container:", config.option.container)
    try:
        iris = IRISContainer(
            config.option.container,
            username="test",
            password="test",
            namespace="TEST",
        )
        iris.start()
        print("Started on port:", iris.get_exposed_port(1972))
        config.option.dburi = iris.get_connection_url()
    except Exception as ex:
        iris = None
        pytest.exit("Failed to start IRIS container: " + str(ex))


def pytest_unconfigure(config):
    global iris
    if iris:
        print("Stopping IRIS container")
        iris.stop()
