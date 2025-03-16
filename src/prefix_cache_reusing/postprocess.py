import re

def add_backticks(s):
    return f"`{s}`"

def post_process(response): 
    response=add_backticks(response)
    response = response.replace('{placeholder}', '<*>')
    response = response.replace('{*}', '<*>') 
    response = response.replace('[<>]', '<*>')
    response = response.replace('<{}>', '<*>') 
    response = response.replace('<*?>', '<*>') 
    response = response.replace('#<*>#', '<*>') 
    response = response.replace('#<*>', '<*>') 
    response = response.replace('<*>#<*>', '<*>') 
    response = response.replace('<*>/<*>', '<*>')
    response = response.replace('<*>.<*>', '<*>') 
    response = response.replace('<**>', '<*>') 
    response = response.replace('\n', '')
    first_backtick_index = response.find('`')
    last_backtick_index = response.rfind('`')
    if first_backtick_index == -1 or last_backtick_index == -1 or first_backtick_index == last_backtick_index:
        tmps = []
    else:
        tmps = response[first_backtick_index: last_backtick_index + 1].split('`')
    for tmp in tmps:
        if tmp.replace(' ','').replace('<*>','') == '':
            tmps.remove(tmp)
    tmp = ''
    if len(tmps) == 1:
        tmp = tmps[0]
    if len(tmps) > 1:
        tmp = max(tmps, key=len)

    template = re.sub(r'\{\{.*?\}\}', '<*>', tmp)
    template = correct_single_template(template)
    if template.replace('<*>', '').replace(' ','') == '':
        template = ''

    return template

def extract_variables(log, template):
    log = re.sub(r'\s+', ' ', log.strip()) 
    pattern_parts = template.split("<*>")
    pattern_parts_escaped = [re.escape(part) for part in pattern_parts]
    regex_pattern = "(.*?)".join(pattern_parts_escaped)
    regex = "^" + regex_pattern + "$"  
    matches = re.search(regex, log)
    if matches:
        return matches.groups()
    else:
        return None
    

def post_process_for_batch_output(response):
    outputs = response.strip('\n').split('\n')
    templates = []
    for output in outputs:
        template = re.sub(r'\{\{.*?\}\}', '<*>', output)
        template = correct_single_template(template)
        if template.replace('<*>', '').strip() == '':
            template = ''
        if template not in templates:
            templates.append(template)
    return templates


def correct_single_template(template, user_strings=None):
    """Apply all rules to process a template.

    DS (Double Space)
    BL (Boolean)
    US (User String)
    DG (Digit)
    PS (Path-like String)
    WV (Word concatenated with Variable)
    DV (Dot-separated Variables)
    CV (Consecutive Variables)

    """

    boolean = {'true', 'false'}
    default_strings = {'null', 'root', 'admin'}
    path_delimiters = {  # reduced set of delimiters for tokenizing for checking the path-like strings
        r'\s', r'\,', r'\!', r'\;', r'\:',
        r'\=', r'\|', r'\"', r'\'',
        r'\[', r'\]', r'\(', r'\)', r'\{', r'\}'
    }
    token_delimiters = path_delimiters.union({  # all delimiters for tokenizing the remaining rules
        r'\.', r'\-', r'\+', r'\@', r'\#', r'\$', r'\%', r'\&',
    })

    if user_strings:
        default_strings = default_strings.union(user_strings)

    # apply DS
    # Note: this is not necessary while postprorcessing
    template = template.strip()
    template = re.sub(r'\s+', ' ', template)

    # apply PS
    p_tokens = re.split('(' + '|'.join(path_delimiters) + ')', template)   # 用path_delimiters做分割
    new_p_tokens = []
    for p_token in p_tokens:  # 匹配路径
        if re.match(r'^(\/[^\/]+)+$', p_token) or all(x in p_token for x in {'<*>', '.', '/'}):
            p_token = '<*>'
        new_p_tokens.append(p_token)
    template = ''.join(new_p_tokens)

    # tokenize for the remaining rules
    tokens = re.split('(' + '|'.join(token_delimiters) + ')', template)  # tokenizing while keeping delimiters
    new_tokens = []
    for token in tokens:
        # apply BL, US
        for to_replace in boolean.union(default_strings):
            if token.lower() == to_replace.lower():
                token = '<*>'

        # apply DG
        # Note: hexadecimal num also appears a lot in the logs
        if re.match(r'^\d+$', token) or re.match(r'\b0[xX][0-9a-fA-F]+\b', token):  #数字或者16进制 0x开头
            token = '<*>'

        # apply WV
        if re.match(r'^[^\s\/]*<\*>[^\s\/]*$', token):  
            # if token != '<*>/<*>':  # need to check this because `/` is not a deliminator
            token = '<*>'

        # collect the result
        new_tokens.append(token)

    # make the template using new_tokens
    template = ''.join(new_tokens)

    for token in template.split(' '):
        if all(x in token for x in {'<*>', '.', ':'}):
            template = template.replace(token, '<*>')

    # Substitute consecutive variables only if separated with any delimiter including "." (DV)
    while True:
        prev = template
        template = re.sub(r'<\*>\.<\*>', '<*>', template)
        if prev == template:
            break

    # Substitute consecutive variables only if not separated with any delimiter including space (CV)
    # NOTE: this should be done at the end
    while True:
        prev = template
        template = re.sub(r'<\*><\*>', '<*>', template)      
        if prev == template:
            break
    # incorrect in HealthApp
    # while "#<*>#" in template:
    #     template = template.replace("#<*>#", "<*>")

    while "<*>:<*>" in template:
        template = template.replace("<*>:<*>", "<*>")   
    while "<*>/<*>" in template:
        template = template.replace("<*>/<*>", "<*>") 

    return template



def matches_template(log, cached_pair):
    reference_log = cached_pair[0]
    template = cached_pair[1]

    # length matters
    if abs(len(log.split()) - len(reference_log.split())) > 1:
        return None

    groups = extract_variables(log, template)
    if not groups:
        return None

    parts = []
    for index, part in enumerate(template.split("<*>")):
        parts.append(part)
        if index < len(groups):
            if groups[index] == '':
                parts.append('')
            else:
                parts.append('<*>')

    return ''.join(parts)


def intersection_count(list1, list2):
    return len(set(list1) & set(list2))


def count_prefix_matches(a, b):
    # a: current  b:target
    a=pair_elements(a)
    b=pair_elements(b)
    match_count=0
    for k_b, v_b in b:
        matched = False
        for i, (k_a, v_a) in enumerate(a):
            if v_b == v_a:  
                match_count += 1 
                a.pop(i) 
                matched = True
                break 
        
        if not matched:
            break
    return match_count

def count_normal_matches(a,b):
    # a:current icl, b:target_icl
    indexes = [1, 3, 5, 7, 9]
    cur_elements = set(a[i] for i in indexes if i < len(a))
    tar_elements = set(b[i] for i in indexes if i < len(b))
    match_count = len(cur_elements & tar_elements)
    return match_count

def match_icl(cur_icl, hist_icl_list):
    mch='prefix'
    if len(hist_icl_list)==0:
        return None,0
    else:
        max_count = 0
        max_intersect_list = None
        
        for hist_icl in hist_icl_list:   
            if mch=='template':
                common_elements_count = count_normal_matches(cur_icl, hist_icl)
            elif mch=='prefix':
                common_elements_count = count_prefix_matches(cur_icl, hist_icl)

            if common_elements_count > max_count:
                max_count = common_elements_count
                max_intersect_list = hist_icl
        if max_count==0:
            return None,0
        else:                    
            return max_intersect_list,max_count

def pair_elements(a):
    return [(a[i], a[i + 1]) for i in range(0, len(a), 2)]



def reorder_and_replace(a, b):
    a=pair_elements(a)
    b=pair_elements(b)
    remaining_a = a[:]
    result = []

    for k_b, v_b in b:
        matched = False
        for i, (k_a, v_a) in enumerate(remaining_a):
            if v_b == v_a: 
                result.append((k_b, v_b)) 
                remaining_a.pop(i) 
                matched = True
                break 

        if not matched:
            break
    result.extend(remaining_a)
    result_flat=[result[0][0],result[0][1],
                 result[1][0],result[1][1],
                 result[2][0],result[2][1],
                 result[3][0],result[3][1],
                 result[4][0],result[4][1]]
    return result_flat

def tokenize(log_content, tokenize_pattern=r'[ ,|]', removeDight=True):
    words = re.split(tokenize_pattern, log_content)
    new_words = []
    vars_count=0
    for word in words:
        if '=' in word and word!='=':
            ws = word.split('=')
            if len(ws) <= 2:
                new_words.append(ws[0])
                vars_count+=1
            else:
                # might be some parameters of a URL 
                pass 

        elif removeDight and re.search(r'\d', word):  
            vars_count+=1
            pass
        elif '/' in word.lower(): 
            vars_count+=1
            pass
        else:
            new_words.append(word)
    new_words = [word for word in new_words if word]   # remove null
    if new_words == []:
        new_words.append(re.sub(r'\d+(\.\d+)?', '0', log_content))
    return new_words,len(new_words),vars_count,len(new_words)+vars_count

