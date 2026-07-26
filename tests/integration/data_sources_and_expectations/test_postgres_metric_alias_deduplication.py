import pandas as pd
import pytest
import sqlalchemy.types as sqltypes

import great_expectations as gx
import great_expectations.expectations as gxe
from great_expectations.data_context import get_context
from tests.integration.test_utils.data_source_config.postgres import (
    PostgresBatchTestSetup,
    PostgreSQLDatasourceTestConfig,
)

pytestmark = pytest.mark.postgresql


def test_postgres_multiple_null_expectations_alias_deduplication():
    """
    Regression test for Issue #10926 / PR #11905.

    Verifies that metric aliasing deduplication functions correctly during SQL compilation.
    Executing multiple expectations that generate identical underlying metrics
    (e.g., column_values.nonnull.unexpected_count) on different columns must not trigger
    a 'Duplicated field name in view schema' OperationalError.
    """

    # 1. Define schema and intentional null values
    df = pd.DataFrame(
        {
            "field1": [1, 2, None, 4],
            "field2": ["a", "b", "c", None],
            "field3": [10.0, 20.0, 30.0, 40.0],
        }
    )

    context = get_context(mode="ephemeral")

    # 2. Initialize Postgres fixture
    batch_setup = PostgresBatchTestSetup(
        config=PostgreSQLDatasourceTestConfig(
            column_types={
                "field1": sqltypes.INTEGER,
                "field2": sqltypes.VARCHAR,
                "field3": sqltypes.FLOAT,
            }
        ),
        data=df,
        extra_data={},
        context=context,
    )

    with batch_setup.batch_test_context() as batch:
        # 3. Construct Expectation Suite
        suite = context.suites.add(gx.ExpectationSuite(name="alias_deduplication_suite"))

        # 4. Add multiple identical expectations on different columns
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="field1"))
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="field2"))
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="field3"))

        # 5. Execute Validation
        # `batch.batch_definition` is a LegacyBatchDefinition and fails
        # ValidationDefinition's pydantic isinstance check. The real fluent
        # BatchDefinition (created via add_batch_definition_whole_table in
        # PostgresBatchTestSetup.make_batch) lives on the data asset instead.
        batch_definition = batch.data_asset.batch_definitions[0]

        validation_definition = context.validation_definitions.add(
            gx.ValidationDefinition(
                name="alias_deduplication_validation",
                data=batch_definition,
                suite=suite,
            )
        )

        result = validation_definition.run()

        # 6. Assert successful execution without alias collision
        assert len(result.results) == 3
