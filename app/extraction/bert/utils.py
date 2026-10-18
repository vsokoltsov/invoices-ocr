def normalize_with_mapping(text):
    normalized = []
    original_positions = []

    previous_was_space = True

    for index, character in enumerate(str(text)):
        if character.isalnum():
            normalized.append(character.lower())
            original_positions.append(index)
            previous_was_space = False

        elif not previous_was_space:
            normalized.append(" ")
            original_positions.append(index)
            previous_was_space = True

    if normalized and normalized[-1] == " ":
        normalized.pop()
        original_positions.pop()

    return "".join(normalized), original_positions