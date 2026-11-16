from enum import Enum
import re

class AudioType(Enum):
    """Type of sound a recording is assumed to contain: lung, heart or both mixed.

    UNKNOWN is returned when the type cannot be determined.
    """
    LUNG = "lung"
    HEART = "heart"
    MIX = "mix"
    UNKNOWN = ""

    @classmethod
    def match_file(cls, dataset: str, group: str) -> "AudioType":
        """Return the AudioType of a file from its dataset and sub-folder.

        Args:
            dataset: dataset name ("hls"/"hls-cmds" or "nih"; case-insensitive).
            group: sub-folder inside the dataset ("lung"/"ls", "heart"/"hs" or "mix"; case-insensitive).
        """
        match dataset.lower():
            case "hls" | "hls-cmds":
                match group.lower():
                    case "lung" | "ls":
                        return cls.LUNG
                    case "heart" | "hs":
                        return cls.HEART
                    case "mix":
                        return cls.MIX
                    case _:
                        return cls.UNKNOWN
            # NIH only contains lung sounds, so its sub-folder does not matter
            case "nih":
                return cls.LUNG
            case _:
                return cls.UNKNOWN

    @classmethod
    def from_filename(cls, file: str) -> "AudioType":
        """Return the AudioType of a file from the first capital letter of its name (H, L or M)."""
        tag = re.search(r"[A-Z]", file).group()
        match tag:
            case "H": return cls.HEART
            case "L": return cls.LUNG
            case "M": return cls.MIX
            case _ : return cls.UNKNOWN
