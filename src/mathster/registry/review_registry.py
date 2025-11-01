"""
Review Registry for Managing Review History and Queries.

This module implements a centralized registry for managing reviews of mathematical
objects in the proof pipeline. The registry provides:
- Storage of review history (all iterations)
- Efficient querying by object ID, status, iteration
- Review comparison and analysis
- Export/import for persistence

The registry uses a singleton pattern to ensure global access while maintaining
referential integrity.

All operations are designed to be total functions (no exceptions for missing data).

Version: 1.0.0
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from mathster.core.review_system import (
    Review,
    ReviewComparison,
    ReviewIssue,
    ReviewSeverity,
    ReviewStatus,
)


# =============================================================================
# REVIEW REGISTRY (Singleton)
# =============================================================================


class ReviewRegistry:
    """
    Centralized registry for managing review history and queries.

    This is a singleton that provides:
    1. Storage: Dict[object_id, List[Review]] (chronological history)
    2. Queries: By status, severity, iteration, source
    3. Comparison: Dual review analysis
    4. Persistence: JSON export/import

    All methods are total functions (return Optional/empty list on missing data).

    Maps to Lean:
        structure ReviewRegistry where
          reviews : HashMap String (List Review)  -- object_id → review history

          def add_review (registry : ReviewRegistry) (r : Review) : ReviewRegistry
          def get_latest (registry : ReviewRegistry) (obj_id : String) : Option Review
          def get_history (registry : ReviewRegistry) (obj_id : String) : List Review
          ...
    """

    _instance: ReviewRegistry | None = None

    def __new__(cls) -> ReviewRegistry:
        """Singleton pattern: ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize registry storage."""
        self._reviews: dict[str, list[Review]] = {}  # object_id → List[Review]
        self._review_by_id: dict[str, Review] = {}  # review_id → Review

    # ==========================================================================
    # STORAGE OPERATIONS
    # ==========================================================================

    def add_review(self, review: Review) -> None:
        """
        Add review to registry.

        Pure function in spirit (registry is singleton, but operation is idempotent).

        Args:
            review: Review to add

        Maps to Lean:
            def add_review (registry : ReviewRegistry) (r : Review) : ReviewRegistry :=
              { registry with
                reviews := registry.reviews.insert r.object_id
                  (registry.reviews.findD r.object_id [] ++ [r])
                review_by_id := registry.review_by_id.insert r.review_id r }
        """
        # Add to object history
        if review.object_id not in self._reviews:
            self._reviews[review.object_id] = []
        self._reviews[review.object_id].append(review)

        # Add to review ID index
        self._review_by_id[review.review_id] = review

    def clear(self) -> None:
        """Clear all reviews (useful for testing)."""
        self._reviews.clear()
        self._review_by_id.clear()

    # ==========================================================================
    # BASIC QUERIES
    # ==========================================================================

    def get_latest_review(self, object_id: str) -> Review | None:
        """
        Total function: Get most recent review for object.

        Returns None if no reviews exist.

        Maps to Lean:
            def get_latest_review (registry : ReviewRegistry) (obj_id : String) : Option Review :=
              match registry.reviews.find? obj_id with
              | none => none
              | some reviews => reviews.getLast?
        """
        if object_id not in self._reviews:
            return None
        if not self._reviews[object_id]:
            return None
        return self._reviews[object_id][-1]

    def get_review_history(self, object_id: str) -> list[Review]:
        """
        Total function: Get complete review history for object (chronological).

        Returns empty list if no reviews exist.

        Maps to Lean:
            def get_review_history (registry : ReviewRegistry) (obj_id : String) : List Review :=
              registry.reviews.findD obj_id []
        """
        return self._reviews.get(object_id, [])

    def get_review_by_id(self, review_id: str) -> Review | None:
        """
        Total function: Get review by its unique ID.

        Returns None if review doesn't exist.

        Maps to Lean:
            def get_review_by_id (registry : ReviewRegistry) (rev_id : String) : Option Review :=
              registry.review_by_id.find? rev_id
        """
        return self._review_by_id.get(review_id)

    def get_iteration_count(self, object_id: str) -> int:
        """
        Total function: Get number of review iterations for object.

        Returns 0 if no reviews exist.

        Maps to Lean:
            def get_iteration_count (registry : ReviewRegistry) (obj_id : String) : Nat :=
              (registry.reviews.findD obj_id []).length
        """
        return len(self._reviews.get(object_id, []))

    # ==========================================================================
    # ISSUE QUERIES
    # ==========================================================================

    def get_unresolved_issues(self, object_id: str) -> list[ReviewIssue]:
        """
        Total function: Get all unresolved issues for object.

        An issue is unresolved if:
        1. It appears in the latest review
        2. It hasn't been addressed in llm_responses

        Returns empty list if no reviews or all issues resolved.

        Maps to Lean:
            def get_unresolved_issues (registry : ReviewRegistry) (obj_id : String) : List ReviewIssue :=
              match registry.get_latest_review obj_id with
              | none => []
              | some review =>
                  let addressed := review.llm_responses.map (fun r => r.issue_id)
                  review.issues.filter (fun i => i.id ∉ addressed)
        """
        latest = self.get_latest_review(object_id)
        if latest is None:
            return []

        addressed_ids = {r.issue_id for r in latest.llm_responses if r.implemented}
        return [i for i in latest.issues if i.id not in addressed_ids]

    def get_blocking_issues(self, object_id: str) -> list[ReviewIssue]:
        """
        Total function: Get all blocking issues for object (from latest review).

        Blocking issues are CRITICAL, MAJOR, or VALIDATION_FAILURE.

        Returns empty list if no reviews or no blocking issues.

        Maps to Lean:
            def get_blocking_issues (registry : ReviewRegistry) (obj_id : String) : List ReviewIssue :=
              match registry.get_latest_review obj_id with
              | none => []
              | some review => review.get_blocking_issues
        """
        latest = self.get_latest_review(object_id)
        if latest is None:
            return []
        return latest.get_blocking_issues()

    def get_issues_by_severity(
        self, object_id: str, severity: ReviewSeverity
    ) -> list[ReviewIssue]:
        """
        Total function: Get issues of specific severity for object (from latest review).

        Returns empty list if no reviews or no matching issues.

        Maps to Lean:
            def get_issues_by_severity
              (registry : ReviewRegistry)
              (obj_id : String)
              (sev : ReviewSeverity)
              : List ReviewIssue :=
              match registry.get_latest_review obj_id with
              | none => []
              | some review => review.get_issues_by_severity sev
        """
        latest = self.get_latest_review(object_id)
        if latest is None:
            return []
        return latest.get_issues_by_severity(severity)

    # ==========================================================================
    # STATUS QUERIES
    # ==========================================================================

    def get_status(self, object_id: str) -> ReviewStatus:
        """
        Total function: Get current review status for object.

        Returns NOT_REVIEWED if no reviews exist.

        Maps to Lean:
            def get_status (registry : ReviewRegistry) (obj_id : String) : ReviewStatus :=
              match registry.get_latest_review obj_id with
              | none => ReviewStatus.not_reviewed
              | some review => review.get_status
        """
        latest = self.get_latest_review(object_id)
        if latest is None:
            return ReviewStatus.NOT_REVIEWED
        return latest.get_status()

    def get_objects_by_status(self, status: ReviewStatus) -> list[str]:
        """
        Total function: Get all object IDs with given status.

        Returns empty list if no objects have that status.

        Maps to Lean:
            def get_objects_by_status (registry : ReviewRegistry) (st : ReviewStatus) : List String :=
              registry.reviews.toList.filter
                (fun (obj_id, reviews) =>
                  match reviews.getLast? with
                  | none => st == ReviewStatus.not_reviewed
                  | some review => review.get_status == st)
                .map (fun (obj_id, _) => obj_id)
        """
        result = []
        for obj_id in self._reviews:
            if self.get_status(obj_id) == status:
                result.append(obj_id)
        return result

    def get_ready_objects(self) -> list[str]:
        """
        Total function: Get all object IDs ready for publication.

        Returns empty list if no ready objects.
        """
        return self.get_objects_by_status(ReviewStatus.READY)

    def get_blocked_objects(self) -> list[str]:
        """
        Total function: Get all object IDs blocked by critical issues.

        Returns empty list if no blocked objects.
        """
        return self.get_objects_by_status(ReviewStatus.BLOCKED)

    # ==========================================================================
    # COMPARISON QUERIES
    # ==========================================================================

    def compare_reviews(self, review_a_id: str, review_b_id: str) -> ReviewComparison | None:
        """
        Total function: Compare two reviews (for dual review analysis).

        Returns None if either review doesn't exist or reviews are for different objects.

        Maps to Lean:
            def compare_reviews
              (registry : ReviewRegistry)
              (id_a id_b : String)
              : Option ReviewComparison :=
              match (registry.get_review_by_id id_a, registry.get_review_by_id id_b) with
              | (some rev_a, some rev_b) =>
                  if rev_a.object_id == rev_b.object_id then
                    some (create_comparison rev_a rev_b)
                  else none
              | _ => none
        """
        review_a = self.get_review_by_id(review_a_id)
        review_b = self.get_review_by_id(review_b_id)

        if review_a is None or review_b is None:
            return None

        if review_a.object_id != review_b.object_id:
            return None

        return self._create_comparison(review_a, review_b)

    def _create_comparison(self, review_a: Review, review_b: Review) -> ReviewComparison:
        """
        Pure function: Create ReviewComparison from two reviews.

        Identifies consensus, discrepancies, and unique issues.

        Algorithm:
        1. Match issues by location + severity (approximate matching)
        2. Issues matching → consensus
        3. Issues with same location but different severity → discrepancy
        4. Unmatched issues → unique
        """
        # Build issue maps by location
        issues_a_by_loc: dict[str, list[ReviewIssue]] = {}
        for issue in review_a.issues:
            if issue.location not in issues_a_by_loc:
                issues_a_by_loc[issue.location] = []
            issues_a_by_loc[issue.location].append(issue)

        issues_b_by_loc: dict[str, list[ReviewIssue]] = {}
        for issue in review_b.issues:
            if issue.location not in issues_b_by_loc:
                issues_b_by_loc[issue.location] = []
            issues_b_by_loc[issue.location].append(issue)

        consensus: list[ReviewIssue] = []
        discrepancies: list[Tuple[ReviewIssue, ReviewIssue]] = []
        unique_to_a: list[ReviewIssue] = []
        unique_to_b: list[ReviewIssue] = []

        matched_b: set[str] = set()

        # Find consensus and discrepancies
        for loc, issues_a in issues_a_by_loc.items():
            if loc not in issues_b_by_loc:
                unique_to_a.extend(issues_a)
                continue

            issues_b = issues_b_by_loc[loc]

            for issue_a in issues_a:
                matched = False
                for issue_b in issues_b:
                    if issue_b.id in matched_b:
                        continue

                    # Check if issues match (same severity = consensus)
                    if issue_a.severity == issue_b.severity:
                        consensus.append(issue_a)  # Use review_a's issue
                        matched_b.add(issue_b.id)
                        matched = True
                        break
                    # Same location, different severity = discrepancy
                    discrepancies.append((issue_a, issue_b))
                    matched_b.add(issue_b.id)
                    matched = True
                    break

                if not matched:
                    unique_to_a.append(issue_a)

        # Find unique to B
        for loc, issues_b in issues_b_by_loc.items():
            for issue_b in issues_b:
                if issue_b.id not in matched_b:
                    if loc not in issues_a_by_loc:
                        unique_to_b.append(issue_b)

        # Compute confidence weights
        confidence_weights = {
            "consensus": 1.0,
            "unique": 0.5,
            "discrepancy": 0.0,  # Requires manual verification
        }

        return ReviewComparison(
            review_a=review_a,
            review_b=review_b,
            consensus_issues=consensus,
            discrepancies=discrepancies,
            unique_to_a=unique_to_a,
            unique_to_b=unique_to_b,
            confidence_weights=confidence_weights,
        )

    # ==========================================================================
    # ITERATION ANALYSIS
    # ==========================================================================

    def get_improvement_trajectory(self, object_id: str) -> list[dict[str, float]]:
        """
        Total function: Get improvement metrics across iterations.

        Returns list of dicts with:
        - iteration: int
        - avg_score: float
        - blocking_count: int
        - total_issues: int

        Returns empty list if no reviews.

        Maps to Lean:
            def get_improvement_trajectory
              (registry : ReviewRegistry)
              (obj_id : String)
              : List (Nat × Float × Nat × Nat) :=
              (registry.get_review_history obj_id).map
                (fun review =>
                  (review.iteration,
                   review.get_average_score,
                   review.blocking_issue_count,
                   review.issues.length))
        """
        history = self.get_review_history(object_id)
        trajectory = []

        for review in history:
            trajectory.append({
                "iteration": review.iteration,
                "avg_score": review.get_average_score(),
                "blocking_count": review.blocking_issue_count,
                "total_issues": len(review.issues),
                "timestamp": review.timestamp.isoformat(),
            })

        return trajectory

    def is_converging(self, object_id: str, min_iterations: int = 2) -> bool:
        """
        Total function: Check if reviews are converging (improving over time).

        Returns False if insufficient iterations or not improving.

        Maps to Lean:
            def is_converging (registry : ReviewRegistry) (obj_id : String) : Bool :=
              let history := registry.get_review_history obj_id
              if history.length < 2 then false
              else
                let last := history.getLast!
                let prev := history.get! (history.length - 2)
                last.is_improvement_over prev
        """
        history = self.get_review_history(object_id)
        if len(history) < min_iterations:
            return False

        # Check if latest is improvement over previous
        latest = history[-1]
        previous = history[-2]
        return latest.is_improvement_over(previous)

    # ==========================================================================
    # PERSISTENCE
    # ==========================================================================

    def export_to_json(self, path: Path) -> None:
        """
        Export registry to JSON file.

        Format:
        {
            "metadata": {...},
            "reviews_by_object": {
                "object-id": [review1.dict(), review2.dict(), ...]
            }
        }
        """
        data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "total_objects": len(self._reviews),
                "total_reviews": len(self._review_by_id),
            },
            "reviews_by_object": {
                obj_id: [r.model_dump(mode="json") for r in reviews]
                for obj_id, reviews in self._reviews.items()
            },
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_from_json(self, path: Path) -> None:
        """
        Load registry from JSON file.

        Clears existing registry before loading.
        """
        self.clear()

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        for reviews_data in data["reviews_by_object"].values():
            for review_data in reviews_data:
                # Convert timestamp strings to datetime
                review_data["timestamp"] = datetime.fromisoformat(review_data["timestamp"])
                for llm_resp in review_data.get("llm_responses", []):
                    llm_resp["timestamp"] = datetime.fromisoformat(llm_resp["timestamp"])
                for val_result in review_data.get("validation_results", []):
                    val_result["timestamp"] = datetime.fromisoformat(val_result["timestamp"])

                review = Review(**review_data)
                self.add_review(review)

    # ==========================================================================
    # STATISTICS
    # ==========================================================================

    def get_statistics(self) -> dict[str, any]:
        """
        Total function: Get registry statistics.

        Returns dict with:
        - total_objects: int
        - total_reviews: int
        - by_status: Dict[ReviewStatus, int]
        - by_source: Dict[ReviewSource, int]
        - avg_iterations: float
        """
        by_status = {}
        for status in ReviewStatus:
            by_status[status.value] = len(self.get_objects_by_status(status))

        by_source = {}
        for review in self._review_by_id.values():
            source = review.source.value
            by_source[source] = by_source.get(source, 0) + 1

        iterations = [len(reviews) for reviews in self._reviews.values()]
        avg_iterations = sum(iterations) / len(iterations) if iterations else 0.0

        return {
            "total_objects": len(self._reviews),
            "total_reviews": len(self._review_by_id),
            "by_status": by_status,
            "by_source": by_source,
            "avg_iterations": avg_iterations,
            "ready_count": by_status.get(ReviewStatus.READY.value, 0),
            "blocked_count": by_status.get(ReviewStatus.BLOCKED.value, 0),
        }


# =============================================================================
# SINGLETON ACCESS
# =============================================================================


def get_review_registry() -> ReviewRegistry:
    """Get the global ReviewRegistry singleton."""
    return ReviewRegistry()
