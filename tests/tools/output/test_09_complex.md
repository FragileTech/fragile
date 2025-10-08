# Test 09: Complex Document

This is a complex test with multiple scenarios.

## Section 1

Some text before equation.

$$
x^2 + y^2 = r^2
$$

More text.

## Section 2

Code with math: `$$formula$$` should not be touched.

```python
def example():
    # $$not_math$$
    return "$$also_not_math$$"
```

## Section 3

Multiple equations:

$$
a = b + c
$$

$$
d = e \cdot f
$$

## Section 4

Properly formatted one:

$$
\int_{a}^{b} f(x) dx
$$

More content here.
