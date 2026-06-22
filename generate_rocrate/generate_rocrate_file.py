from rocrate.rocrate import ROCrate
from rocrate.model.contextentity import ContextEntity

crate = ROCrate()

geo_entity = ContextEntity(crate, identifier="https://www.geonames.org/10858713/university-of-nottingham.html", properties={
    "@type": "https://schema.org/GeoCoordinates",
    "latitude": "52.941686822260635",
    "longitude": "-1.1871061808863315",
    "address": {
        "@id": "https://www.geonames.org/10858713/university-of-nottingham.html",
        "@type": "https://schema.org/PostalAddress",
        "addressLocality": "University Park, Nottingham",
        "addressRegion": "Nottinghamshire",
        "addressCountry": "UK",
        "streetAddress": "Biodiscovery Institute",
        "postalCode": "NG7 2RD"
    }
})

crate.add(geo_entity)

institution_id = "https://ror.org/01ee9ar58"

institute_entity = ContextEntity(crate, identifier=institution_id, properties = {
    "@type": "Organization",
    "name": "University of Nottingham",
    "url": "https://www.nottingham.ac.uk",
    "location": geo_entity,
})

crate.add(institute_entity)

crate.root_dataset["about"] = institute_entity

crate.write("tre_metadata_crate")