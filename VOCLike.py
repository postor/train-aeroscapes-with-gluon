from gluoncv.data import VOCSegmentation


class VOCLike(VOCSegmentation):
    NUM_CLASS = 2
    BASE_DIR = 'aeroscapes'
    @property
    def classes(self):
        """Category names."""
        return ('background',
                'person',
                'bike',
                'car',
                'drone',
                'boat',
                'animal',
                'obstacle',
                'construction',
                'vegetation',
                'road',
                'sky')
