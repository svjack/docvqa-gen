import text2text as t2t
from tqdm import tqdm

eng_context = '''
a petition dated 14th August, 2o05 for administrative review with the Director General, Inland Revenue Department.The Director General vide his Order dated 17th January, 2006 rejected the said petition. SNPL thereafter filed an appeal to the Revenue Tribunal, which refused to entertain the appeal in the absence of a pre-deposit of the entire
'''
t2t.Handler(["{} [SEP] {}".format(eng_context, "SNPL")], src_lang='en').question()
'''
#### output format
[('Whose song does I like?', 'HuiLin Chen')]
'''


#t2t.Handler(["{} [SEP] {}".format(eng_context, "SNPL")], src_lang='en').question()
def gen_ask_by_span_en(t2t ,sent, span):
    assert type(sent) == type("")
    #assert type(span) == type("")
    if type(span) == type(""):
        span = [span]
    if not span:
        return []
    req = []
    for ele in tqdm(span):
        output = t2t.Handler(["{} [SEP] {}".format(sent, ele)], src_lang='en').question()
        assert type(output) == type([])
        assert bool(output)
        output = output[0][0]
        assert type(output) == type("")
        req.append(output)
    return req
