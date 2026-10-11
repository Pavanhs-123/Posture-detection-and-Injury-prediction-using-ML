from mediapipe.tasks.python import vision
import pkgutil
print([name for _, name, _ in pkgutil.iter_modules(vision.__path__)])
