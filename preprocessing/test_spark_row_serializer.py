import unittest
import base64
import json
import pickle
import datetime
from unittest.mock import patch, MagicMock
from pyspark.sql import Row

# Import the module containing the SparkRowSerializer
from preprocessing.kafka import SparkRowSerializer


class TestSparkRowSerializer(unittest.TestCase):
    """Test cases for the SparkRowSerializer class."""

    def setUp(self):
        """Setup the test environment before each test."""
        self.serializer = SparkRowSerializer()

        # Create sample Row objects for testing
        self.simple_dict = {"id": 1, "name": "Test User", "active": True}
        self.simple_row = Row(**self.simple_dict)

        # Create a more complex Row with nested structures
        self.complex_dict = {
            "id": 123,
            "user": {"name": "Jane Doe", "email": "jane@example.com"},
            "tags": ["python", "spark", "kafka"],
            "metrics": {"visits": 42, "conversions": 3.14},
            "timestamp": datetime.datetime(2023, 1, 1, 12, 0, 0),
        }
        # Note: For a proper test, you'd need to create a proper nested Row structure
        # This is simplified for testing purposes
        self.complex_row = Row(**self.complex_dict)

        # Sample metadata
        self.metadata = {
            "source": "test_application",
            "version": "1.0.0",
            "priority": "high",
        }

    def test_serialize_simple_row(self):
        """Test serializing a simple Row object."""
        # Test serialization without metadata
        serialized_data, headers = self.serializer.serialize_message(self.simple_row)

        # Verify it's base64 encoded
        self.assertTrue(isinstance(serialized_data, bytes))

        # Verify headers
        self.assertTrue(any(h.get("key") == "schema" for h in headers))
        self.assertTrue(
            any(
                h.get("key") == "format" and h.get("value") == b"pickle_base64"
                for h in headers
            )
        )

        # Find and decode the schema header
        schema_header = next(h for h in headers if h.get("key") == "schema")
        schema_info = json.loads(schema_header.get("value").decode("utf-8"))

        # Check schema contents
        self.assertEqual(schema_info["type"], "spark_row")
        self.assertTrue("timestamp" in schema_info)

        # Decode and unpickle the data to verify contents
        pickle_data = base64.b64decode(serialized_data)
        row_dict = pickle.loads(pickle_data)

        # Verify the original data was preserved
        self.assertEqual(row_dict, self.simple_dict)

    def test_serialize_with_metadata(self):
        """Test serializing a Row with metadata."""
        serialized_data, headers = self.serializer.serialize_message(
            self.simple_row, self.metadata
        )

        # Find and decode the schema header
        schema_header = next(h for h in headers if h.get("key") == "schema")
        schema_info = json.loads(schema_header.get("value").decode("utf-8"))

        # Verify metadata was included
        self.assertTrue("metadata" in schema_info)
        self.assertEqual(schema_info["metadata"], self.metadata)

    def test_serialize_complex_row(self):
        """Test serializing a Row with complex nested structure."""
        serialized_data, headers = self.serializer.serialize_message(self.complex_row)

        # Decode and unpickle
        pickle_data = base64.b64decode(serialized_data)
        row_dict = pickle.loads(pickle_data)

        # Verify nested structures were preserved
        self.assertEqual(row_dict["id"], self.complex_dict["id"])
        self.assertEqual(row_dict["user"], self.complex_dict["user"])
        self.assertEqual(row_dict["tags"], self.complex_dict["tags"])
        self.assertEqual(row_dict["metrics"], self.complex_dict["metrics"])
        self.assertEqual(row_dict["timestamp"], self.complex_dict["timestamp"])

    def test_serialize_dict_instead_of_row(self):
        """Test serializing a dictionary instead of a Row."""
        serialized_data, headers = self.serializer.serialize_message(self.simple_dict)

        # Decode and unpickle
        pickle_data = base64.b64decode(serialized_data)
        row_dict = pickle.loads(pickle_data)

        # Verify the data was preserved
        self.assertEqual(row_dict, self.simple_dict)

    def test_deserialize_simple_message(self):
        """Test deserializing a simple message."""
        # First serialize a message
        serialized_data, headers = self.serializer.serialize_message(self.simple_row)

        # Convert headers to the format expected by deserialize_message
        headers_tuples = [(h["key"], h["value"]) for h in headers]

        # Now deserialize it
        row_dict, metadata = self.serializer.deserialize_message(
            serialized_data, headers_tuples
        )

        # Verify the deserialized data matches the original
        self.assertEqual(row_dict, self.simple_dict)
        self.assertTrue("timestamp" in metadata)

    def test_deserialize_with_metadata(self):
        """Test deserializing a message with metadata."""
        # Serialize with metadata
        serialized_data, headers = self.serializer.serialize_message(
            self.simple_row, self.metadata
        )

        # Convert headers to the format expected by deserialize_message
        headers_tuples = [(h["key"], h["value"]) for h in headers]

        # Deserialize
        row_dict, metadata = self.serializer.deserialize_message(
            serialized_data, headers_tuples
        )

        # Verify metadata was preserved
        for key, value in self.metadata.items():
            self.assertEqual(metadata[key], value)

    def test_deserialize_complex_message(self):
        """Test deserializing a complex message."""
        # Serialize complex data
        serialized_data, headers = self.serializer.serialize_message(self.complex_row)

        # Convert headers
        headers_tuples = [(h["key"], h["value"]) for h in headers]

        # Deserialize
        row_dict, metadata = self.serializer.deserialize_message(
            serialized_data, headers_tuples
        )

        # Verify nested structures were preserved
        self.assertEqual(row_dict["id"], self.complex_dict["id"])
        self.assertEqual(row_dict["user"], self.complex_dict["user"])
        self.assertEqual(row_dict["tags"], self.complex_dict["tags"])
        self.assertEqual(row_dict["metrics"], self.complex_dict["metrics"])
        self.assertEqual(row_dict["timestamp"], self.complex_dict["timestamp"])

    def test_deserialize_with_no_headers(self):
        """Test deserializing with missing headers."""
        # Serialize data
        serialized_data, _ = self.serializer.serialize_message(self.simple_row)

        # Deserialize with no headers
        row_dict, metadata = self.serializer.deserialize_message(serialized_data, None)

        # Verify data was still deserialized
        self.assertEqual(row_dict, self.simple_dict)
        # Metadata should be empty
        self.assertEqual(metadata, {})

    def test_deserialize_invalid_data(self):
        """Test deserializing invalid data."""
        # Create invalid serialized data
        invalid_data = base64.b64encode(b"This is not valid pickle data")

        # Attempt to deserialize
        with self.assertRaises(Exception):
            self.serializer.deserialize_message(invalid_data, None)

    @patch("logging.Logger.debug")
    def test_logging_during_serialization(self, mock_debug):
        """Test that logging occurs during serialization."""
        self.serializer.serialize_message(self.simple_row)

        # Verify debug logging occurred
        mock_debug.assert_any_call("Serializing PySpark Row")
        # The second call contains variable data, so we just check it was called again
        self.assertTrue(mock_debug.call_count >= 2)

    @patch("logging.Logger.debug")
    def test_logging_during_deserialization(self, mock_debug):
        """Test that logging occurs during deserialization."""
        # First serialize to create valid data
        serialized_data, headers = self.serializer.serialize_message(self.simple_row)
        headers_tuples = [(h["key"], h["value"]) for h in headers]

        # Reset mock
        mock_debug.reset_mock()

        # Now deserialize
        self.serializer.deserialize_message(serialized_data, headers_tuples)

        # Verify debug logging occurred
        mock_debug.assert_any_call("Deserializing PySpark Row message")
        self.assertTrue(mock_debug.call_count >= 2)

    @patch("logging.Logger.error")
    def test_logging_during_deserialization_error(self, mock_error):
        """Test error logging during deserialization failure."""
        # Create invalid data
        invalid_data = base64.b64encode(b"This is not valid pickle data")

        # Attempt to deserialize, catching the exception
        try:
            self.serializer.deserialize_message(invalid_data, None)
        except:
            pass

        # Verify error was logged
        mock_error.assert_called_once()
        self.assertTrue("Error deserializing Row:" in mock_error.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
