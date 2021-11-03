def gen_meows():
    current_message = "Meow"
    while True:
        yield current_message
        current_message = f"{current_message} {current_message}"


generator = gen_meows()
for i in range(10):
    print(next(generator))
