from .common import printd

def eval_fstring(template, **args):
    return template.format(**args)

def eval_jinja2(template, **args):
    from jinja2 import Template
    return  Template(template).render(args)


def eval_template(template, **args):
    if '{{' and '}}' in template:
        return eval_jinja2(template, **args)
    else:
        printd(1, '**Warning**: fstrings in prompt templates will be deprecated. Please use jinja2 template format instead.')
        return eval_fstring(template, **args)


