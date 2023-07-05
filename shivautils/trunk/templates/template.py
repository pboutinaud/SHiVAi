from jinja2 import Template

file = "templates/a.html"

f = open('test.html', 'w', encoding='utf-8')

tm = Template(
              """<!DOCTYPE html>
                 <html lang="en">
                 <head>
                    <meta charset="UTF-8">
                    <title>My template</title>
                 </head>
                 <body>
                 <div class="test">
                    <iframe src="{{ pa }}" width="500px" height="500px" frameborder="0"></iframe>
                 </div>
                 </body>
                 </html>"""
                 )

msg = tm.render(pa=file)

f.write(msg)
f.close()
