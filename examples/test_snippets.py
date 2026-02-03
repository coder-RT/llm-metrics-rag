"""
Example Code Snippets for Testing

These are sample code snippets that would be provided to candidates
in a hackathon scenario. The candidates can use Cline (or other AI tools)
to help them complete/debug these snippets.

When snippet_grounded mode is ON, the LLM will be constrained to work
only with these provided snippets.
"""

# =============================================================================
# SNIPPET 1: Binary Search Implementation
# =============================================================================
# Candidates need to implement binary search algorithm

def binary_search(arr: list, target: int) -> int:
    """
    Search for target in a sorted array using binary search.
    
    Args:
        arr: A sorted list of integers
        target: The value to search for
        
    Returns:
        Index of target if found, -1 otherwise
        
    Example:
        >>> binary_search([1, 2, 3, 4, 5], 3)
        2
        >>> binary_search([1, 2, 3, 4, 5], 6)
        -1
    """
    # TODO: Implement binary search
    # Hint: Use left and right pointers
    pass


# =============================================================================
# SNIPPET 2: Two Sum Problem
# =============================================================================
# Classic interview problem

def two_sum(nums: list, target: int) -> list:
    """
    Find two numbers that add up to target.
    
    Args:
        nums: List of integers
        target: Target sum
        
    Returns:
        List of two indices whose values add up to target
        
    Example:
        >>> two_sum([2, 7, 11, 15], 9)
        [0, 1]
    """
    # TODO: Implement two sum
    # Hint: Consider using a hash map for O(n) solution
    pass


# =============================================================================
# SNIPPET 3: Linked List Node with Bug
# =============================================================================
# This snippet has a bug that candidates need to find

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse_linked_list(head: ListNode) -> ListNode:
    """
    Reverse a singly linked list.
    
    Args:
        head: Head of the linked list
        
    Returns:
        New head of the reversed list
        
    Example:
        1 -> 2 -> 3 -> None
        becomes
        3 -> 2 -> 1 -> None
    """
    prev = None
    current = head
    
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        # BUG: Missing line to move current forward
        # current = next_node  # <-- This line is missing!
    
    return prev


# =============================================================================
# SNIPPET 4: Rate Limiter (More Complex)
# =============================================================================

class RateLimiter:
    """
    Implement a rate limiter using sliding window algorithm.
    
    The rate limiter should allow at most `max_requests` requests
    within a `window_seconds` time window.
    """
    
    def __init__(self, max_requests: int, window_seconds: int):
        """
        Initialize the rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []  # List of timestamps
    
    def allow_request(self, timestamp: float) -> bool:
        """
        Check if a request should be allowed.
        
        Args:
            timestamp: Current timestamp in seconds
            
        Returns:
            True if request is allowed, False if rate limited
        """
        # TODO: Implement sliding window rate limiting
        # 1. Remove timestamps outside the window
        # 2. Check if we're under the limit
        # 3. If allowed, record this request
        pass


# =============================================================================
# TEST CASES (for verification)
# =============================================================================

def test_binary_search():
    assert binary_search([1, 2, 3, 4, 5], 3) == 2
    assert binary_search([1, 2, 3, 4, 5], 1) == 0
    assert binary_search([1, 2, 3, 4, 5], 5) == 4
    assert binary_search([1, 2, 3, 4, 5], 6) == -1
    assert binary_search([], 1) == -1
    print("binary_search: All tests passed!")


def test_two_sum():
    assert two_sum([2, 7, 11, 15], 9) == [0, 1]
    assert two_sum([3, 2, 4], 6) == [1, 2]
    assert two_sum([3, 3], 6) == [0, 1]
    print("two_sum: All tests passed!")


if __name__ == "__main__":
    print("Test snippets loaded. Run tests with:")
    print("  test_binary_search()")
    print("  test_two_sum()")
