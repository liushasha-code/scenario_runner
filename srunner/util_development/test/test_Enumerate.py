
a = ['a', 'b', 'c', 'd', 'e', 'f']

b = ['kkk', 'jjj']

c = [a, b]

d = ['test']
for item in c:
    d = d+item



d = a.extend(b)

e = a.append(b)

print('d')

num = 2

b.append(a[num:])

print('d')


for ids, item in enumerate(a):
    print('id', ids)
    print('item: ', item)
    print('')


print('d')
