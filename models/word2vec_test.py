from gensim.models import Word2Vec
# sentences只需要是一个可迭代对象就可以
sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
model = Word2Vec(sentences, min_count=1)  # 执行这一句的时候就是在训练模型了
