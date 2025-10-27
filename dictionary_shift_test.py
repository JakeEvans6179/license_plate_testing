from collections import deque, defaultdict
'''
testing_d = {1:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}

value = 1
#new_value = 16
iteration_list = [0,1,2,3,4,5]
for i in iteration_list:
    new_value = i
    if value not in testing_d:
        testing_d[value] = []  # create list to hold the images with track id as the key


    if len(testing_d[value]) <= 14:
        testing_d[value].append(new_value)  # append the image details onto the list if less than 15 elements
        print("added onto list")

    else:
        testing_d[value].pop(0)
        print("removed from list")

        testing_d[value].append(new_value)

    print(testing_d)

print(testing_d)
print(len(testing_d[value]))
'''
frame = 100
frame_values = defaultdict(int)

default_d = defaultdict(lambda: deque(maxlen=15)) #creates rolling dictionary holding 15 of the latest frames and another element, if key is called that isn't on the dictionary it calls lambda function

default_d[5].append('NEW-PLATE')
frame_values[5]= frame


#track id 5 already exists so append
frame += 1
default_d[5].append('NEW-PLATE')
frame_values[5] = frame

#new deque created for id 2
frame += 1
default_d[2].append('XYZ-789')
frame_values[2] = frame

print(default_d)

print("Default_d key 5 first element:", default_d[5][0])
print("Default_d key 5 second element:", default_d[5][1])
print("Default_d key 2 first element:", default_d[2][0])

print(frame_values)

print(list(frame_values.keys()))
print(list(frame_values.values()))

test = 'hello'

print(list(test))







