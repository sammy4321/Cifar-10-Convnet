import numpy as np
import tensorflow as tf

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

read_from_file=unpickle("test_batch")
data=read_from_file['data']
labels=read_from_file['labels']

data=np.array(data)
labels=np.array(labels)
#number_of_batches=len(data)/100

x=tf.placeholder('float',[None,32,32,1])
y=tf.placeholder('float')

def convolutional_neural_network(data):
    #data=tf.reshape(data,shape=[-1,32,32,1])

    conv1=tf.contrib.layers.conv2d(inputs=data,num_outputs=32,kernel_size=[5,5],stride=1,rate=1,activation_fn=tf.nn.relu,padding='SAME',weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float32),biases_initializer=tf.zeros_initializer(),trainable=True)
    conv1=tf.contrib.layers.max_pool2d(inputs=conv1,kernel_size=[2,2],stride=[2,2],padding='SAME')
    conv1=tf.contrib.layers.batch_norm(conv1,center=True, scale=True,is_training=True)
    #conv1=tf.nn.dropout(conv1,0.5)
    
    conv2=tf.contrib.layers.conv2d(inputs=conv1,num_outputs=64,kernel_size=[5,5],stride=1,rate=1,activation_fn=tf.nn.relu,padding='SAME',weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float32),biases_initializer=tf.zeros_initializer(),trainable=True)
    conv2=tf.contrib.layers.max_pool2d(inputs=conv2,kernel_size=[2,2],stride=[2,2],padding='SAME')
    conv2=tf.contrib.layers.batch_norm(conv2,center=True, scale=True,is_training=True)
    #conv2=tf.nn.dropout(conv2,0.5)
 
    
    fc=tf.reshape(conv2,[-1,3*8*8*64])
    fc=tf.contrib.layers.fully_connected(fc,num_outputs=1024,activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float32),biases_initializer=tf.zeros_initializer(),trainable=True)
    fc=tf.contrib.layers.batch_norm(fc,center=True, scale=True,is_training=True)


    output=tf.contrib.layers.fully_connected(fc,num_outputs=10,activation_fn=None,weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float32),biases_initializer=tf.zeros_initializer(),trainable=True)

    return output
def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    
    hm_epochs = 10
    cost_list=[]
    accuracy_list=[]
    no_epochs_list=[]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Training Begins...")

        for epoch in range(hm_epochs):
            epoch_loss = 0
            print('Epoch : ',epoch)
            
            read_from_file=unpickle("data_batch_1")
            print("Data batch - 1")
            data=read_from_file['data']
            labels=read_from_file['labels']
            data=np.array(data)
            labels=np.array(labels)
            number_of_batches=len(data)/100


            for i in range(number_of_batches):
            	print('Epoch : ',epoch," Data batch - 1 , Batch number : ",i,' Out of : ',number_of_batches)
            	epoch_x=data[100*i:100*(i+1)]
            	yy=labels[100*i:100*(i+1)]
            	n_values=np.max(yy)+1
            	epoch_y=np.eye(n_values)[yy]
            	#Epoch_y=np.zeros((100,10))
            	#Epoch_y[np.arange(100),epoch_y]=1
            	
                #epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x.reshape((-1,32,32,1)), y: epoch_y})
                #print("Epoch : ",i," Cost : ",)
                epoch_loss += c
                del epoch_x
                del epoch_y
                del yy

            read_from_file=unpickle("data_batch_2")
            print("Data batch - 2")
            data=read_from_file['data']
            labels=read_from_file['labels']
            data=np.array(data)
            labels=np.array(labels)
            number_of_batches=len(data)/100


            for i in range(number_of_batches):
            	print('Epoch : ',epoch," Data batch - 2 , Batch number : ",i,' Out of : ',number_of_batches)
            	epoch_x=data[100*i:100*(i+1)]
            	yy=labels[100*i:100*(i+1)]
            	n_values=np.max(yy)+1
            	epoch_y=np.eye(n_values)[yy]
            	#Epoch_y=np.zeros((100,10))
            	#Epoch_y[np.arange(100),epoch_y]=1
            	
                #epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x.reshape((-1,32,32,1)), y: epoch_y})
                #print("Epoch : ",i," Cost : ",)
                epoch_loss += c
                del epoch_x
                del epoch_y
                del yy    


            read_from_file=unpickle("data_batch_3")
            print("Data batch - 3")
            data=read_from_file['data']
            labels=read_from_file['labels']
            data=np.array(data)
            labels=np.array(labels)
            number_of_batches=len(data)/100


            for i in range(number_of_batches):
            	print('Epoch : ',epoch," Data batch - 3 , Batch number : ",i,' Out of : ',number_of_batches)
            	epoch_x=data[100*i:100*(i+1)]
            	yy=labels[100*i:100*(i+1)]
            	n_values=np.max(yy)+1
            	epoch_y=np.eye(n_values)[yy]
            	#Epoch_y=np.zeros((100,10))
            	#Epoch_y[np.arange(100),epoch_y]=1
            	
                #epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x.reshape((-1,32,32,1)), y: epoch_y})
                #print("Epoch : ",i," Cost : ",)
                epoch_loss += c
                del epoch_x
                del epoch_y
                del yy
            #cost_list.append(epoch_loss)
            #no_epochs_list.append(epoch)
            #acc=accuracy.eval({x:mnist.test.images, y:mnist.test.labels})
            #acc=acc*100
            #accuracy_list.append(acc)
            
            


            read_from_file=unpickle("data_batch_4")
            print("Data batch - 4")
            data=read_from_file['data']
            labels=read_from_file['labels']
            data=np.array(data)
            labels=np.array(labels)
            number_of_batches=len(data)/100


            for i in range(number_of_batches):
            	print('Epoch : ',epoch," Data batch - 4 , Batch number : ",i,' Out of : ',number_of_batches)
            	epoch_x=data[100*i:100*(i+1)]
            	yy=labels[100*i:100*(i+1)]
            	n_values=np.max(yy)+1
            	epoch_y=np.eye(n_values)[yy]
            	#Epoch_y=np.zeros((100,10))
            	#Epoch_y[np.arange(100),epoch_y]=1
            	
                #epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x.reshape((-1,32,32,1)), y: epoch_y})
                #print("Epoch : ",i," Cost : ",)
                epoch_loss += c
                del epoch_x
                del epoch_y
                del yy
            


            read_from_file=unpickle("data_batch_5")
            print("Data batch - 5")
            data=read_from_file['data']
            labels=read_from_file['labels']
            data=np.array(data)
            labels=np.array(labels)
            number_of_batches=len(data)/100


            for i in range(number_of_batches):
            	print('Epoch : ',epoch," Data batch - 5 , Batch number : ",i,' Out of : ',number_of_batches)
            	epoch_x=data[100*i:100*(i+1)]
            	yy=labels[100*i:100*(i+1)]
            	n_values=np.max(yy)+1
            	epoch_y=np.eye(n_values)[yy]
            	#Epoch_y=np.zeros((100,10))
            	#Epoch_y[np.arange(100),epoch_y]=1
            	
                #epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x.reshape((-1,32,32,1)), y: epoch_y})
                #print("Epoch : ",i," Cost : ",)
                epoch_loss += c
                del epoch_x
                del epoch_y
                del yy
            print('Epoch : ',epoch,' Cost : ',epoch_loss)    

            #print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss,'Accuracy : ',acc)

        

        print("Running test batch....")
        read_from_file=unpickle("test_batch")
        data=read_from_file['data']
        labels=read_from_file['labels']
        data=np.array(data)
        labels=np.array(labels)
        number_of_batches=len(data)/100
        #test_x=data.reshape((-1,32,32,1))
        #n_values=np.max(labels)+1
        #test_y=np.eye(n_values)[labels]
        acc =0
        for i in range(number_of_batches):
        	print('Testing : ',i,' Out of : ',number_of_batches)
        	epoch_x=data[100*i:100*(i+1)]
        	yy=labels[100*i:100*(i+1)]
        	n_values=np.max(yy)+1
        	epoch_y=np.eye(n_values)[yy]
        	acc = acc+accuracy.eval({x: epoch_x.reshape((-1,32,32,1)), y: epoch_y})
        	del epoch_x
        	del epoch_y
        	del yy



        print('Accuracy : ',acc)

        

        

train_neural_network(x)
