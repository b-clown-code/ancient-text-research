import unittest
from catalog_corpus import classify, representation, header_metadata
from catalog_documents import tt_metadata

class CatalogTests(unittest.TestCase):
    def test_modern_translations_and_metadata_are_not_primary(self):
        self.assertEqual(classify('papyri-info','Translations/test.xml'),'translation_reference')
        self.assertEqual(classify('papyri-info','APIS/test.xml'),'metadata_or_support')
        self.assertEqual(classify('suttacentral-bilara','root/en/site/test.json'),'modern_editorial')
        self.assertEqual(classify('suttacentral-bilara','root/pli/ms/test.json'),'primary_text_candidate')

    def test_representation_group_preserves_source_document(self):
        self.assertEqual(representation('aed-tei','files/abc_hiero.xml'),representation('aed-tei','files/abc.xml'))
        self.assertNotEqual(representation('aed-tei','files/abc.xml'),representation('aed-tei','files/def.xml'))

    def test_date_contexts_remain_separate(self):
        data=b'<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><publicationStmt><date>2020</date></publicationStmt><sourceDesc><origDate notBefore="1200" notAfter="1300"/></sourceDesc></fileDesc></teiHeader><text>body</text></TEI>'
        m=header_metadata(data)
        self.assertEqual(m['header_status'],'parsed')
        self.assertIn('publicationStmt',m['dates'][0]['context'])
        self.assertEqual(m['dates'][1]['attributes']['notAfter'],'1300')
        self.assertNotIn('composition_date',m)

    def test_entity_is_not_resolved(self):
        m=header_metadata(b'<teiHeader><title>&private;</title></teiHeader>')
        self.assertEqual(m['header_status'],'parse_error')

    def test_treetagger_repeated_attributes_are_preserved(self):
        attrs,repeated=tt_metadata('<meta language="Coptic" people="Paul" people="Peter" origDate="800" title="A &amp; B">')
        self.assertEqual(attrs['title'],'A & B')
        self.assertEqual(attrs['origdate'],'800')
        self.assertEqual(repeated['people'],['Paul','Peter'])

if __name__=='__main__': unittest.main()
