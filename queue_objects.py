from enum import Enum

class QueueEventTypes(Enum):
  PLAY_VIDEO = 1
  FREEZE_FRAME = 2
  FREEZE_SUBTITLE = 3
  FREEZE_BOTH = 4
  SHOW_SUBTITLE = 5
  KILL_FRAME_PROCESSING = 6
  PROCESS_IMAGES = 7
  PROCESSING_DONE = 8

class QueueEvent():
  def __init__(self, type, payload):
    self.type = type
    self.payload = payload