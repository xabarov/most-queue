from most_queue.theory.utils.moments import convert_raw_to_central


def test_convert_raw_to_central_empty():
    """Test conversion with empty input."""
    assert convert_raw_to_central([]) == []


def test_convert_raw_to_central_single_moment():
    """Test conversion with only the first raw moment (mean)."""
    raw_moments = [5.0]
    expected = [5.0]
    assert convert_raw_to_central(raw_moments) == expected


def test_convert_raw_to_central_two_moments():
    """Test conversion up to variance."""
    raw_moments = [2.0, 5.0]  # E[X], E[X^2]
    expected = [2.0, 1.0]  # 0, E[X^2] - (E[X])^2 = 5 - 4 = 1
    assert convert_raw_to_central(raw_moments) == expected


def test_convert_raw_to_central_three_moments():
    """Test conversion including skewness."""
    raw_moments = [1.0, 2.0, 6.0]  # E[X], E[X^2], E[X^3]
    expected = [
        1.0,  # First central moment is zero
        1.0,  # Variance: 2 - 1^2 = 1
        2.0,  # Skewness: 6 - 3*1*2 + 2*1^3 = 6 - 6 + 2 = 2
    ]
    assert convert_raw_to_central(raw_moments) == expected


def test_convert_raw_to_central_four_moments():
    """Test conversion including kurtosis."""
    raw_moments = [1.0, 2.0, 6.0, 24.0]  # E[X], E[X^2], E[X^3], E[X^4]
    expected = [
        1.0,  # First central moment is zero
        1.0,  # Variance: 2 - 1^2 = 1
        2.0,  # Skewness: 6 - 3*1*2 + 2*1^3 = 2
        9.0,  # Kurtosis: 24 - 4*1*6 + 6*1^2*2 - 3*1^4 = 24 - 24 + 12 - 3 = 9
    ]
    assert convert_raw_to_central(raw_moments) == expected
