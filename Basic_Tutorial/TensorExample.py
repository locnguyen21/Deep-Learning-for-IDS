import tensorflow as tf

a = tf.constant(5)
b = tf.constant(2)
c = tf.constant(3)

d = tf.multiply(a,b)
e = tf.add(c,d)
f = tf.subtract(d,e)

sess = tf.Session()
outs = sess.run(f)
print("outs = {}".format(outs))

out = sess.run(d)
sess.close()
print(out)
#print(type(out)) <class 'numpy.int32'>
#graph có thể parameterized (tham số hóa) mà không cần constant bằng cách
#chấp nhận external inputs -----> placeholder
#placeholder cung cấp giá trị ở đằng sau

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

#<class 'tensorflow.python.framework.ops.Tensor'>
#print(type(a))

adder_node = a + b
#print(adder_node) #Tensor("add_1:0", dtype=float32)
sess = tf.Session()
add = sess.run(adder_node,{a:[1,3],b:[2,4]})
#print(type(add)) #<class 'numpy.ndarray'>
print(add)

w = tf.Variable([.3],tf.float32)
b = tf.Variable([-.3],tf.float32)
x = tf.placeholder(tf.float32)

linear_model = w*x + b #tensor type
init = tf.global_variables_initializer()
sess = tf.Session()
g = sess.run(init)
#print(g) #None
g = sess.run(linear_model,{x:[1,2,3,4]})
print(g) #numpy array

#calculate loss function
#y: giá trị mong đợi, giá trị đúng
y = tf.placeholder(tf.float32)
#linear_model - y là khoảng cách trên đồ thị giũa 2 điểm đoán và điểm thật
#square: bình phương
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas) #reduce sum là tổng sum
squared = sess.run(squared_deltas,{x:[1,2,3,4],y:[0,-1,-2,-3]})
print(squared)
#lost_function = sum của squared
loss_function = sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]})
print(loss_function) #numpy array

#Tối ưu hóa optimized để loss function thấp hơn