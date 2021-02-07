from hieudl import FaceRecognizer

recognizer = FaceRecognizer()
# recognizer.register('Hieu')
# recognizer.register('Obama', video='obama.mp4')
# recognizer.register('Biden', video='biden.mp4')
recognizer.start_standalone_app()
