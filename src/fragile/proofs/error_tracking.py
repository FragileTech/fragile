"""
Error Tracking and Logging for Extract-then-Enrich Pipeline.

This module provides comprehensive error tracking and logging infrastructure
for the LLM-based mathematical paper extraction pipeline:

- ErrorLogger: Centralized logging with structured error capture
- ValidationReport: Results summary with metrics
- ErrorSummary: Aggregated error statistics
- Integration with EnrichmentError and ValidationResult

Maps to Lean:
    structure ErrorLog where
      timestamp : DateTime
      error_type : ErrorType
      message : String
      entity_id : Option String

    structure ValidationReport where
      is_valid : Bool
      total_entities : Nat
      successful : Nat
      failed : Nat
      errors : List ErrorLog
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from fragile.proofs.orchestration import (
    EnrichmentError,
    EnrichmentStatus,
    ErrorType,
)


# =============================================================================
# ERROR LOG ENTRY
# =============================================================================


class ErrorLogEntry(BaseModel):
    """
    A single error log entry.

    Captures all information about an error for later analysis and debugging.

    Examples:
        >>> entry = ErrorLogEntry(
        ...     timestamp=datetime.now(),
        ...     error_type=ErrorType.PARSE_FAILURE,
        ...     message="Failed to parse LaTeX",
        ...     entity_id="raw-eq-001",
        ...     context={"latex": "\\frac{1}{x}"}
        ... )

    Maps to Lean:
        structure ErrorLogEntry where
          timestamp : DateTime
          error_type : ErrorType
          message : String
          entity_id : Option String
          context : HashMap String Any
          severity : String
    """

    timestamp: datetime = Field(default_factory=datetime.now)
    error_type: ErrorType
    message: str
    entity_id: Optional[str] = None
    entity_type: Optional[str] = None  # "theorem", "definition", etc.
    context: Dict[str, Any] = Field(default_factory=dict)
    severity: str = Field(default="error")  # "warning", "error", "critical"
    recoverable: bool = Field(default=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "error_type": self.error_type.value,
            "message": self.message,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "context": self.context,
            "severity": self.severity,
            "recoverable": self.recoverable,
        }


# =============================================================================
# ERROR SUMMARY
# =============================================================================


class ErrorSummary(BaseModel):
    """
    Aggregated error statistics for a pipeline run.

    Provides high-level metrics and breakdowns by error type.

    Examples:
        >>> summary = ErrorSummary(
        ...     total_errors=15,
        ...     by_type={ErrorType.PARSE_FAILURE: 10, ErrorType.REFERENCE_UNRESOLVED: 5},
        ...     by_entity_type={"theorem": 8, "definition": 7}
        ... )

    Maps to Lean:
        structure ErrorSummary where
          total_errors : Nat
          by_type : HashMap ErrorType Nat
          by_entity_type : HashMap String Nat
          recoverable_count : Nat
          critical_count : Nat
    """

    total_errors: int = 0
    by_type: Dict[ErrorType, int] = Field(default_factory=dict)
    by_entity_type: Dict[str, int] = Field(default_factory=dict)
    by_severity: Dict[str, int] = Field(default_factory=dict)
    recoverable_count: int = 0
    critical_count: int = 0

    def add_error(self, entry: ErrorLogEntry) -> None:
        """Update statistics with a new error entry."""
        self.total_errors += 1

        # Count by type
        if entry.error_type not in self.by_type:
            self.by_type[entry.error_type] = 0
        self.by_type[entry.error_type] += 1

        # Count by entity type
        if entry.entity_type:
            if entry.entity_type not in self.by_entity_type:
                self.by_entity_type[entry.entity_type] = 0
            self.by_entity_type[entry.entity_type] += 1

        # Count by severity
        if entry.severity not in self.by_severity:
            self.by_severity[entry.severity] = 0
        self.by_severity[entry.severity] += 1

        # Update recoverable/critical counts
        if entry.recoverable:
            self.recoverable_count += 1
        if entry.severity == "critical":
            self.critical_count += 1

    def get_most_common_error_type(self) -> Optional[ErrorType]:
        """Get the most frequently occurring error type."""
        if not self.by_type:
            return None
        return max(self.by_type.items(), key=lambda x: x[1])[0]


# =============================================================================
# VALIDATION REPORT
# =============================================================================


class ValidationReport(BaseModel):
    """
    Comprehensive validation report for a pipeline run.

    Summarizes extraction, enrichment, and validation results with detailed
    error tracking and metrics.

    Examples:
        >>> report = ValidationReport(
        ...     document_id="03_cloning",
        ...     total_entities_extracted=50,
        ...     successful_enrichments=42,
        ...     failed_enrichments=8
        ... )
        >>> report.get_success_rate()
        0.84

    Maps to Lean:
        structure ValidationReport where
          document_id : String
          timestamp : DateTime
          is_valid : Bool
          total_entities_extracted : Nat
          successful_enrichments : Nat
          failed_enrichments : Nat
          errors : List ErrorLogEntry
          error_summary : ErrorSummary
    """

    # Metadata
    document_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    pipeline_stage: str = Field(
        default="enrichment", description="Stage: 'extraction', 'enrichment', 'validation'"
    )

    # Overall status
    is_valid: bool = Field(default=True)

    # Entity counts
    total_entities_extracted: int = 0
    successful_enrichments: int = 0
    failed_enrichments: int = 0
    skipped_enrichments: int = 0

    # Error tracking
    errors: List[ErrorLogEntry] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    error_summary: ErrorSummary = Field(default_factory=ErrorSummary)

    # Timing
    duration_seconds: Optional[float] = None

    def add_error(
        self,
        error_type: ErrorType,
        message: str,
        entity_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        severity: str = "error",
        recoverable: bool = True,
    ) -> None:
        """
        Add an error to the report.

        Args:
            error_type: Classification of error
            message: Human-readable error message
            entity_id: Entity temp ID or label
            entity_type: Type of entity (e.g., "theorem", "definition")
            context: Additional context dict
            severity: "warning", "error", or "critical"
            recoverable: Whether error is recoverable
        """
        entry = ErrorLogEntry(
            error_type=error_type,
            message=message,
            entity_id=entity_id,
            entity_type=entity_type,
            context=context or {},
            severity=severity,
            recoverable=recoverable,
        )
        self.errors.append(entry)
        self.error_summary.add_error(entry)

        # Update overall validity
        if severity == "critical":
            self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def get_success_rate(self) -> float:
        """Calculate enrichment success rate."""
        total = self.successful_enrichments + self.failed_enrichments
        if total == 0:
            return 1.0
        return self.successful_enrichments / total

    def get_error_rate(self) -> float:
        """Calculate error rate."""
        return 1.0 - self.get_success_rate()

    def has_critical_errors(self) -> bool:
        """Check if any critical errors occurred."""
        return self.error_summary.critical_count > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "document_id": self.document_id,
            "timestamp": self.timestamp.isoformat(),
            "pipeline_stage": self.pipeline_stage,
            "is_valid": self.is_valid,
            "total_entities_extracted": self.total_entities_extracted,
            "successful_enrichments": self.successful_enrichments,
            "failed_enrichments": self.failed_enrichments,
            "skipped_enrichments": self.skipped_enrichments,
            "success_rate": self.get_success_rate(),
            "error_rate": self.get_error_rate(),
            "total_errors": len(self.errors),
            "total_warnings": len(self.warnings),
            "error_summary": {
                "by_type": {k.value: v for k, v in self.error_summary.by_type.items()},
                "by_entity_type": self.error_summary.by_entity_type,
                "by_severity": self.error_summary.by_severity,
                "recoverable_count": self.error_summary.recoverable_count,
                "critical_count": self.error_summary.critical_count,
            },
            "duration_seconds": self.duration_seconds,
        }


# =============================================================================
# ERROR LOGGER
# =============================================================================


class ErrorLogger:
    """
    Centralized error logger for the extraction pipeline.

    Provides structured logging with automatic file output, console output,
    and error aggregation.

    Examples:
        >>> logger = ErrorLogger("03_cloning", log_dir="logs/")
        >>> logger.log_error(
        ...     ErrorType.PARSE_FAILURE,
        ...     "Failed to parse LaTeX equation",
        ...     entity_id="raw-eq-001",
        ...     entity_type="equation"
        ... )
        >>> logger.log_warning("Missing equation label")
        >>> report = logger.get_report()
        >>> logger.save_report()

    Maps to Lean:
        structure ErrorLogger where
          document_id : String
          log_file_path : String
          report : ValidationReport

          def log_error : ErrorType → String → IO Unit
          def log_warning : String → IO Unit
          def get_report : IO ValidationReport
    """

    def __init__(
        self,
        document_id: str,
        log_dir: Optional[str] = None,
        console_output: bool = True,
        file_output: bool = True,
    ):
        """
        Initialize ErrorLogger.

        Args:
            document_id: Document being processed
            log_dir: Directory for log files (default: ./logs/)
            console_output: Whether to log to console
            file_output: Whether to log to file
        """
        self.document_id = document_id
        self.console_output = console_output
        self.file_output = file_output

        # Create log directory
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            self.log_dir = Path("./logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set up Python logging
        self.logger = logging.getLogger(f"extraction.{document_id}")
        self.logger.setLevel(logging.INFO)

        # Console handler
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if file_output:
            log_file = self.log_dir / f"{document_id}_extraction.log"
            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Validation report
        self.report = ValidationReport(document_id=document_id)

        # Start time
        self.start_time = datetime.now()

    def log_error(
        self,
        error_type: ErrorType,
        message: str,
        entity_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        severity: str = "error",
        recoverable: bool = True,
    ) -> None:
        """
        Log an error.

        Args:
            error_type: Classification of error
            message: Human-readable error message
            entity_id: Entity temp ID or label
            entity_type: Type of entity
            context: Additional context
            severity: "warning", "error", or "critical"
            recoverable: Whether error is recoverable
        """
        # Add to report
        self.report.add_error(
            error_type=error_type,
            message=message,
            entity_id=entity_id,
            entity_type=entity_type,
            context=context,
            severity=severity,
            recoverable=recoverable,
        )

        # Log to Python logger
        log_msg = f"[{error_type.value}] {message}"
        if entity_id:
            log_msg += f" (entity: {entity_id})"

        if severity == "critical":
            self.logger.critical(log_msg)
        elif severity == "error":
            self.logger.error(log_msg)
        else:
            self.logger.warning(log_msg)

    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        self.report.add_warning(message)
        self.logger.warning(message)

    def log_info(self, message: str) -> None:
        """Log an informational message."""
        self.logger.info(message)

    def log_enrichment_error(self, error: EnrichmentError) -> None:
        """
        Log an EnrichmentError.

        Convenience method for logging EnrichmentError instances.

        Args:
            error: EnrichmentError to log
        """
        self.log_error(
            error_type=error.error_type,
            message=error.message,
            entity_id=error.entity_id,
            context=error.context,
            severity="error",
            recoverable=True,
        )

    def increment_extracted(self) -> None:
        """Increment total entities extracted counter."""
        self.report.total_entities_extracted += 1

    def increment_successful(self) -> None:
        """Increment successful enrichments counter."""
        self.report.successful_enrichments += 1

    def increment_failed(self) -> None:
        """Increment failed enrichments counter."""
        self.report.failed_enrichments += 1

    def increment_skipped(self) -> None:
        """Increment skipped enrichments counter."""
        self.report.skipped_enrichments += 1

    def update_status(self, status: EnrichmentStatus) -> None:
        """Update counters based on enrichment status."""
        if status == EnrichmentStatus.COMPLETED:
            self.increment_successful()
        elif status == EnrichmentStatus.FAILED:
            self.increment_failed()
        elif status == EnrichmentStatus.SKIPPED:
            self.increment_skipped()

    def get_report(self) -> ValidationReport:
        """
        Get the current validation report.

        Calculates duration and returns report.

        Returns:
            ValidationReport with all errors and metrics
        """
        # Update duration
        self.report.duration_seconds = (datetime.now() - self.start_time).total_seconds()
        return self.report

    def save_report(self, filename: Optional[str] = None) -> Path:
        """
        Save the validation report to JSON file.

        Args:
            filename: Optional custom filename (default: {document_id}_report.json)

        Returns:
            Path to the saved report file
        """
        if filename is None:
            filename = f"{self.document_id}_report.json"

        report_path = self.log_dir / filename
        report_dict = self.get_report().to_dict()

        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=2)

        self.log_info(f"Validation report saved to: {report_path}")
        return report_path

    def print_summary(self) -> None:
        """Print a summary of the extraction/enrichment results."""
        report = self.get_report()

        print("\n" + "=" * 70)
        print(f"VALIDATION REPORT: {report.document_id}")
        print("=" * 70)
        print(f"Status: {'✓ VALID' if report.is_valid else '✗ INVALID'}")
        print(f"Duration: {report.duration_seconds:.2f}s")
        print()
        print(f"Total entities extracted: {report.total_entities_extracted}")
        print(f"Successful enrichments:   {report.successful_enrichments}")
        print(f"Failed enrichments:       {report.failed_enrichments}")
        print(f"Skipped enrichments:      {report.skipped_enrichments}")
        print(f"Success rate:             {report.get_success_rate():.1%}")
        print()
        print(f"Total errors:   {len(report.errors)}")
        print(f"Total warnings: {len(report.warnings)}")
        print(f"Critical errors: {report.error_summary.critical_count}")
        print()

        if report.error_summary.by_type:
            print("Errors by type:")
            for error_type, count in sorted(
                report.error_summary.by_type.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  - {error_type.value}: {count}")
            print()

        if report.error_summary.by_entity_type:
            print("Errors by entity type:")
            for entity_type, count in sorted(
                report.error_summary.by_entity_type.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                print(f"  - {entity_type}: {count}")
            print()

        print("=" * 70 + "\n")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_logger_for_document(
    document_id: str, log_dir: Optional[str] = None
) -> ErrorLogger:
    """
    Create an ErrorLogger for a document.

    Convenience function for creating loggers.

    Args:
        document_id: Document being processed
        log_dir: Optional log directory

    Returns:
        Configured ErrorLogger instance
    """
    return ErrorLogger(document_id, log_dir=log_dir)


def merge_reports(reports: List[ValidationReport]) -> ValidationReport:
    """
    Merge multiple validation reports into one.

    Useful for combining section-level reports into a document-level report.

    Args:
        reports: List of ValidationReport instances

    Returns:
        Merged ValidationReport
    """
    if not reports:
        raise ValueError("Cannot merge empty list of reports")

    # Use first report as base
    merged = ValidationReport(document_id=reports[0].document_id)

    for report in reports:
        # Aggregate metrics
        merged.total_entities_extracted += report.total_entities_extracted
        merged.successful_enrichments += report.successful_enrichments
        merged.failed_enrichments += report.failed_enrichments
        merged.skipped_enrichments += report.skipped_enrichments

        # Merge errors
        merged.errors.extend(report.errors)
        merged.warnings.extend(report.warnings)

        # Update error summary
        for entry in report.errors:
            merged.error_summary.add_error(entry)

        # Update overall validity
        if not report.is_valid:
            merged.is_valid = False

    return merged
