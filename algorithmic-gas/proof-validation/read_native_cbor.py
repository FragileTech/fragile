"""Read retained native CBOR without optional packages, keeping only the first step.

Other steps are parsed and skipped; the caller separately checks the compressed
artifact SHA256. This decoder does not confer archive validation or proof credit.
"""

import struct


class FirstStepReader:
    def __init__(self, data, prefix_only=False):
        self.data = memoryview(data)
        self.position = 0
        self.prefix_only = prefix_only

    def take(self, count):
        end = self.position + count
        if end > len(self.data):
            msg = "Truncated CBOR"
            raise ValueError(msg)
        out = self.data[self.position : end]
        self.position = end
        return out

    def header(self):
        head = self.take(1)[0]
        major, extra = head >> 5, head & 31
        if extra < 24:
            return major, extra, extra
        if extra in {24, 25, 26, 27}:
            payload = self.take(1 << (extra - 24))
            return major, extra, payload
        msg = "Unsupported indefinite or reserved CBOR encoding"
        raise ValueError(msg)

    def value(self, depth=0, keep=True):
        if depth > 64:
            msg = "CBOR nesting exceeds native archive limit"
            raise ValueError(msg)
        major, extra, payload = self.header()
        number = payload if isinstance(payload, int) else int.from_bytes(payload, "big")
        if major == 7:
            if extra in {25, 26, 27}:
                return (
                    struct.unpack({25: ">e", 26: ">f", 27: ">d"}[extra], payload)[0]
                    if keep
                    else None
                )
            if extra in {20, 21, 22}:
                return {20: False, 21: True, 22: None}[extra] if keep else None
            msg = "Unsupported CBOR simple value"
            raise ValueError(msg)
        if major in {0, 1}:
            return number if major == 0 else -number - 1
        if major in {2, 3}:
            data = self.take(number)
            if not keep:
                return None
            return bytes(data) if major == 2 else bytes(data).decode()
        if major == 4:
            if keep:
                return [self.value(depth + 1) for _ in range(number)]
            # Native scalar fields are homogeneous floating arrays. Checking
            # every fixed-width header permits a bulk skip without weakening
            # validation of their definite CBOR item count.
            if number and self.position < len(self.data):
                width = {0xF9: 3, 0xFA: 5, 0xFB: 9}.get(self.data[self.position])
                if width is not None:
                    end = self.position + width * number
                    if end <= len(self.data):
                        headers = bytes(self.data[self.position : end : width])
                        if headers.count(headers[:1]) == number:
                            self.position = end
                            return None
            for _ in range(number):
                self.value(depth + 1, keep=False)
            return None
        if major == 5:
            result = {}
            for _ in range(number):
                key = self.value(depth + 1, keep=keep)
                if keep and depth == 0 and key == "steps":
                    result[key] = self.first_array(depth + 1)
                    if self.prefix_only:
                        return result
                else:
                    value = self.value(depth + 1, keep=keep)
                    if keep:
                        result[key] = value
            return result if keep else None
        if major == 6:
            return self.value(depth + 1, keep=keep)
        msg = "Unsupported CBOR major type"
        raise ValueError(msg)

    def first_array(self, depth):
        major, _, payload = self.header()
        if major != 4:
            msg = "Native steps must be a definite CBOR array"
            raise ValueError(msg)
        count = payload if isinstance(payload, int) else int.from_bytes(payload, "big")
        first = [self.value(depth + 1)] if count else []
        if self.prefix_only:
            return first
        for _ in range(max(0, count - 1)):
            self.value(depth + 1, keep=False)
        return first


def first_native_step(data, *, verify_remainder=True):
    """Decode the selected prefix, optionally validating all remaining CBOR.

    Prefix extraction does not assert that the unparsed remainder is valid.
    Callers must keep its scope distinct from full native archive validation.
    """
    if len(data) > 256 * 1024 * 1024:
        msg = "Decoded native archive exceeds its 256MiB read budget"
        raise ValueError(msg)
    reader = FirstStepReader(data, prefix_only=not verify_remainder)
    result = reader.value()
    if verify_remainder and reader.position != len(data):
        msg = "Trailing native CBOR data"
        raise ValueError(msg)
    return result
