from enum import Enum


class AudioType(Enum):
    LUNG = "lung"
    HEART = "heart"
    MIX = "mix"
    UNKNOWN = ""

    @classmethod
    def match_file(cls, dataset: str, group: str) -> "AudioType":
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
            case "nih":
                return cls.LUNG
            case _:
                return cls.UNKNOWN
