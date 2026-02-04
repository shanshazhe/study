"""Advanced merge/join: keys, validation, suffixes, and indicator."""

import pandas as pd


def main() -> None:
    users = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "name": ["alice", "bob", "chris"],
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": [10, 11, 12, 13],
            "user_id": [1, 1, 2, 4],
            "amount": [100, 150, 80, 200],
        }
    )

    # Validate one-to-many relationship
    merged = users.merge(
        orders,
        on="user_id",
        how="left",
        validate="one_to_many",
        indicator=True,
        suffixes=("_user", "_order"),
    )
    print("merge with validation and indicator:\n", merged, sep="")

    # Merge on different column names
    left = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
    right = pd.DataFrame({"user_id": [2, 3], "score": [88, 91]})
    print("\nmerge on different keys:\n", left.merge(right, left_on="id", right_on="user_id", how="left"), sep="")


if __name__ == "__main__":
    main()
