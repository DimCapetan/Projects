import pypdf

super_pdf = '/Users/dim/Downloads/merged.pdf'
path = '/Users/dim/Downloads'

# Load the watermark
wtr = pypdf.PdfReader('/Users/dim/Downloads/wtr.pdf').pages[0]
writer = pypdf.PdfWriter(clone_from = super_pdf)
for page in writer.pages:
    page.merge_page(wtr, over = False)

writer.write(f'{path}/updated.pdf')

print('All Done!')