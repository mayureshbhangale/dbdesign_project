select count(*) from title t,cast_info ci,movie_companies mc,movie_info_idx mi_idx where t.id=ci.movie_id and t.id=mc.movie_id and t.id=mi_idx.movie_id and t.production_year < 1898 and ci.role_id < 9 and mi_idx.info_type_id < 112;
select count(*) from title t,cast_info ci,movie_keyword mk,movie_info_idx mi_idx where t.id=ci.movie_id and t.id=mi_idx.movie_id and t.id=mk.movie_id and t.production_year = 1904 and mi_idx.info_type_id < 102 and t.kind_id < 2;
select count(*) from title t,movie_keyword mk,movie_info mi,movie_info_idx mi_idx where t.id=mi.movie_id and t.id=mi_idx.movie_id and t.id=mk.movie_id and mi_idx.info_type_id < 103 and mk.keyword_id > 103056 and t.kind_id < 2;
select count(*) from title t,cast_info ci,movie_info mi,movie_info_idx mi_idx where t.id=ci.movie_id and t.id=mi.movie_id and t.id=mi_idx.movie_id and t.kind_id < 6 and ci.role_id < 4 and t.production_year = 1950;
select count(*) from title t,cast_info ci,movie_companies mc,movie_info_idx mi_idx where t.id=ci.movie_id and t.id=mc.movie_id and t.id=mi_idx.movie_id and ci.role_id = 1 and t.kind_id > 6 and mc.company_type_id = 2;
select count(*) from title t,cast_info ci,movie_keyword mk,movie_info_idx mi_idx where t.id=ci.movie_id and t.id=mi_idx.movie_id and t.id=mk.movie_id and mk.keyword_id = 95308 and mi_idx.info_type_id < 107 and t.kind_id > 3;
select count(*) from title t,cast_info ci,movie_companies mc,movie_info mi where t.id=ci.movie_id and t.id=mc.movie_id and t.id=mi.movie_id and ci.role_id = 10 and t.production_year < 1964 and t.kind_id = 1;
select count(*) from title t,cast_info ci,movie_keyword mk,movie_info mi where t.id=ci.movie_id and t.id=mi.movie_id and t.id=mk.movie_id and t.production_year > 1983 and mi.info_type_id > 29 and mk.keyword_id < 2293;
select count(*) from title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx where t.id=mc.movie_id and t.id=mi.movie_id and t.id=mi_idx.movie_id and mc.company_type_id = 2 and mc.company_id > 35183 and mi.info_type_id = 108;
select count(*) from title t,movie_companies mc,movie_keyword mk,movie_info_idx mi_idx where t.id=mc.movie_id and t.id=mi_idx.movie_id and t.id=mk.movie_id and mi_idx.info_type_id < 104 and mc.company_id < 167404 and t.kind_id > 3;
select count(*) from title t,cast_info ci,movie_companies mc,movie_keyword mk where t.id=ci.movie_id and t.id=mc.movie_id and t.id=mk.movie_id and mk.keyword_id > 61168 and ci.role_id = 2 and mc.company_id > 222409;
select count(*) from title t,movie_keyword mk,movie_info mi,movie_info_idx mi_idx where t.id=mi.movie_id and t.id=mi_idx.movie_id and t.id=mk.movie_id and t.kind_id > 6 and mk.keyword_id > 51846 and t.production_year > 1987;
select count(*) from title t,movie_keyword mk,movie_info mi,movie_info_idx mi_idx where t.id=mi.movie_id and t.id=mi_idx.movie_id and t.id=mk.movie_id and mk.keyword_id > 19895 and t.production_year = 1917 and t.kind_id < 2;
select count(*) from title t,cast_info ci,movie_companies mc,movie_keyword mk where t.id=ci.movie_id and t.id=mc.movie_id and t.id=mk.movie_id and mc.company_type_id > 1 and mk.keyword_id > 3490 and t.production_year > 1896;
select count(*) from title t,cast_info ci,movie_keyword mk,movie_info mi where t.id=ci.movie_id and t.id=mi.movie_id and t.id=mk.movie_id and mk.keyword_id = 80770 and t.production_year < 1942 and ci.person_id > 1093745;
select count(*) from title t,cast_info ci,movie_info mi,movie_info_idx mi_idx where t.id=ci.movie_id and t.id=mi.movie_id and t.id=mi_idx.movie_id and t.kind_id < 6 and ci.person_id < 2334300 and mi.info_type_id > 88;
select count(*) from title t,cast_info ci,movie_companies mc,movie_keyword mk where t.id=ci.movie_id and t.id=mc.movie_id and t.id=mk.movie_id and t.kind_id < 6 and t.production_year > 1929 and mc.company_type_id > 1;
select count(*) from title t,movie_keyword mk,movie_info mi,movie_info_idx mi_idx where t.id=mi.movie_id and t.id=mi_idx.movie_id and t.id=mk.movie_id and t.production_year = 1966 and mi_idx.info_type_id > 105 and mk.keyword_id < 112742;
select count(*) from title t,movie_companies mc,movie_keyword mk,movie_info_idx mi_idx where t.id=mc.movie_id and t.id=mi_idx.movie_id and t.id=mk.movie_id and t.production_year > 1924 and mc.company_type_id > 1 and mc.company_id < 189261;
select count(*) from title t,cast_info ci,movie_keyword mk,movie_info mi where t.id=ci.movie_id and t.id=mi.movie_id and t.id=mk.movie_id and t.production_year = 1951 and mi.info_type_id = 94 and ci.person_id > 2931888;
select count(*) from title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx where t.id=mc.movie_id and t.id=mi.movie_id and t.id=mi_idx.movie_id and t.kind_id < 3 and mc.company_type_id = 1 and mi_idx.info_type_id < 102;
select count(*) from title t,cast_info ci,movie_companies mc,movie_info_idx mi_idx where t.id=ci.movie_id and t.id=mc.movie_id and t.id=mi_idx.movie_id and ci.person_id < 4046435 and t.kind_id = 2 and t.production_year > 1900;
select count(*) from title t,cast_info ci,movie_companies mc,movie_info mi where t.id=ci.movie_id and t.id=mc.movie_id and t.id=mi.movie_id and t.production_year = 1974 and ci.role_id > 4 and ci.person_id > 520513;
select count(*) from title t,cast_info ci,movie_companies mc,movie_keyword mk where t.id=ci.movie_id and t.id=mc.movie_id and t.id=mk.movie_id and mc.company_type_id = 1 and t.kind_id > 2 and ci.role_id < 11;
select count(*) from title t,cast_info ci,movie_companies mc,movie_info_idx mi_idx where t.id=ci.movie_id and t.id=mc.movie_id and t.id=mi_idx.movie_id and mi_idx.info_type_id < 108 and t.production_year < 1906 and mc.company_id > 166204;
select count(*) from title t,cast_info ci,movie_keyword mk,movie_info mi where t.id=ci.movie_id and t.id=mi.movie_id and t.id=mk.movie_id and t.production_year < 1945 and mi.info_type_id < 15 and t.kind_id = 3;
select count(*) from title t,cast_info ci,movie_companies mc,movie_keyword mk where t.id=ci.movie_id and t.id=mc.movie_id and t.id=mk.movie_id and mk.keyword_id = 95205 and ci.role_id < 7 and mc.company_id < 134739;
select count(*) from title t,movie_companies mc,movie_keyword mk,movie_info mi where t.id=mc.movie_id and t.id=mi.movie_id and t.id=mk.movie_id and mc.company_id > 171729 and t.production_year = 1931 and mi.info_type_id = 96;
select count(*) from title t,cast_info ci,movie_keyword mk,movie_info_idx mi_idx where t.id=ci.movie_id and t.id=mi_idx.movie_id and t.id=mk.movie_id and mk.keyword_id < 2801 and t.production_year > 1886 and ci.role_id = 10;
select count(*) from title t,cast_info ci,movie_keyword mk,movie_info_idx mi_idx where t.id=ci.movie_id and t.id=mi_idx.movie_id and t.id=mk.movie_id and mk.keyword_id < 5162 and t.production_year > 1935 and ci.role_id = 5;
select count(*) from title t,cast_info ci,movie_keyword mk,movie_info mi where t.id=ci.movie_id and t.id=mi.movie_id and t.id=mk.movie_id and mk.keyword_id > 124605 and mi.info_type_id < 36 and t.production_year = 2010;
select count(*) from title t,cast_info ci,movie_companies mc,movie_keyword mk where t.id=ci.movie_id and t.id=mc.movie_id and t.id=mk.movie_id and t.kind_id > 4 and mc.company_type_id > 1 and mc.company_id < 66100;
select count(*) from title t,cast_info ci,movie_keyword mk,movie_info mi where t.id=ci.movie_id and t.id=mi.movie_id and t.id=mk.movie_id and ci.role_id = 2 and mi.info_type_id > 14 and ci.person_id > 1543855;
select count(*) from title t,cast_info ci,movie_companies mc,movie_keyword mk where t.id=ci.movie_id and t.id=mc.movie_id and t.id=mk.movie_id and ci.person_id > 1655763 and ci.role_id > 5 and t.production_year < 1946;
select count(*) from title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx where t.id=mc.movie_id and t.id=mi.movie_id and t.id=mi_idx.movie_id and mi.info_type_id < 21 and mi_idx.info_type_id > 100 and mc.company_id = 94999;
select count(*) from title t,movie_companies mc,movie_keyword mk,movie_info_idx mi_idx where t.id=mc.movie_id and t.id=mi_idx.movie_id and t.id=mk.movie_id and mc.company_type_id = 2 and t.kind_id > 6 and mc.company_id < 149503;
select count(*) from title t,cast_info ci,movie_companies mc,movie_keyword mk where t.id=ci.movie_id and t.id=mc.movie_id and t.id=mk.movie_id and mc.company_type_id = 1 and ci.role_id > 3 and mc.company_id < 89224;
select count(*) from title t,cast_info ci,movie_keyword mk,movie_info mi where t.id=ci.movie_id and t.id=mi.movie_id and t.id=mk.movie_id and ci.role_id > 5 and t.production_year > 1906 and mk.keyword_id < 130458;
select count(*) from title t,movie_keyword mk,movie_info mi,movie_info_idx mi_idx where t.id=mi.movie_id and t.id=mi_idx.movie_id and t.id=mk.movie_id and mk.keyword_id > 94751 and mi.info_type_id < 42 and t.kind_id < 4;
select count(*) from title t,cast_info ci,movie_keyword mk,movie_info mi where t.id=ci.movie_id and t.id=mi.movie_id and t.id=mk.movie_id and t.production_year < 1895 and t.kind_id < 3 and ci.person_id < 1450704;
select count(*) from title t,cast_info ci,movie_info mi,movie_info_idx mi_idx where t.id=ci.movie_id and t.id=mi.movie_id and t.id=mi_idx.movie_id and mi_idx.info_type_id > 105 and ci.person_id = 2212605 and t.production_year > 1886;
select count(*) from title t,movie_companies mc,movie_keyword mk,movie_info mi where t.id=mc.movie_id and t.id=mi.movie_id and t.id=mk.movie_id and mi.info_type_id = 9 and mc.company_id < 138688 and t.production_year > 1950;
select count(*) from title t,movie_companies mc,movie_keyword mk,movie_info_idx mi_idx where t.id=mc.movie_id and t.id=mi_idx.movie_id and t.id=mk.movie_id and mi_idx.info_type_id < 110 and t.kind_id = 2 and t.production_year > 2012;
select count(*) from title t,cast_info ci,movie_companies mc,movie_info mi where t.id=ci.movie_id and t.id=mc.movie_id and t.id=mi.movie_id and ci.person_id > 3594860 and t.production_year > 1882 and ci.role_id > 5;
select count(*) from title t,cast_info ci,movie_companies mc,movie_keyword mk where t.id=ci.movie_id and t.id=mc.movie_id and t.id=mk.movie_id and t.production_year > 1984 and mc.company_id = 44800 and ci.person_id < 344959;
select count(*) from title t,cast_info ci,movie_companies mc,movie_keyword mk where t.id=ci.movie_id and t.id=mc.movie_id and t.id=mk.movie_id and t.production_year < 1953 and mk.keyword_id > 125626 and ci.role_id < 4;
select count(*) from title t,cast_info ci,movie_companies mc,movie_keyword mk where t.id=ci.movie_id and t.id=mc.movie_id and t.id=mk.movie_id and mk.keyword_id = 71626 and t.kind_id < 5 and mc.company_type_id = 1;
select count(*) from title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx where t.id=mc.movie_id and t.id=mi.movie_id and t.id=mi_idx.movie_id and mi.info_type_id < 50 and mc.company_type_id = 1 and mc.company_id < 139894;
select count(*) from title t,cast_info ci,movie_keyword mk,movie_info mi where t.id=ci.movie_id and t.id=mi.movie_id and t.id=mk.movie_id and t.production_year = 1930 and mi.info_type_id < 101 and mk.keyword_id > 14699;
select count(*) from title t,movie_companies mc,movie_keyword mk,movie_info_idx mi_idx where t.id=mc.movie_id and t.id=mi_idx.movie_id and t.id=mk.movie_id and mc.company_type_id < 2 and mi_idx.info_type_id < 100 and mk.keyword_id < 30734;
select count(*) from title t,movie_companies mc,movie_keyword mk,movie_info mi where t.id=mc.movie_id and t.id=mi.movie_id and t.id=mk.movie_id and mc.company_type_id > 1 and mk.keyword_id < 64614 and mi.info_type_id > 46;
select count(*) from title t,cast_info ci,movie_companies mc,movie_info_idx mi_idx where t.id=ci.movie_id and t.id=mc.movie_id and t.id=mi_idx.movie_id and ci.person_id < 3942550 and t.kind_id < 5 and mc.company_type_id < 2;
select count(*) from title t,cast_info ci,movie_companies mc,movie_info mi where t.id=ci.movie_id and t.id=mc.movie_id and t.id=mi.movie_id and mi.info_type_id = 7 and ci.person_id < 3507413 and ci.role_id > 4;
select count(*) from title t,movie_keyword mk,movie_info mi,movie_info_idx mi_idx where t.id=mi.movie_id and t.id=mi_idx.movie_id and t.id=mk.movie_id and mi.info_type_id < 33 and mk.keyword_id < 124130 and t.production_year = 1903;
select count(*) from title t,cast_info ci,movie_companies mc,movie_info_idx mi_idx where t.id=ci.movie_id and t.id=mc.movie_id and t.id=mi_idx.movie_id and ci.role_id = 9 and mc.company_id > 181975 and t.production_year < 1968;
select count(*) from title t,movie_companies mc,movie_info mi,movie_info_idx mi_idx where t.id=mc.movie_id and t.id=mi.movie_id and t.id=mi_idx.movie_id and mc.company_type_id = 1 and t.kind_id > 1 and t.production_year = 2006;