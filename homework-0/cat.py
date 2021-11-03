class Cat:

    def __init__(self, name):
        self.name = name

    def greet(self, other):
        print(f'Hello I am {self.name}! I see you are also a cool fluffy kitty {other.name},'
              f' let’s together purr at the human, so that they shall give us food.')
