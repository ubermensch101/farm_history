drop table if exists
    public.filtered_amravati_pahani;
create table
    public.filtered_amravati_pahani
as
    select
        *
    from
        public.amravati_pahani;

-- Dagdagad surveys only
-- delete from
--     public.filtered_amravati_pahani as p
-- where not exists (
--     select
--         1
--     from
--         s2.filtered_farmplots_amravati as f
--     where
--         st_contains(f.geom, p.geom)
-- );

-- Area of survey should match farmplot area
delete from
    public.filtered_amravati_pahani as p
where not exists (
    select
        1
    from
        s2.filtered_farmplots_amravati as f
    where
        st_contains(f.geom, p.geom)
    and
        abs(st_area(f.geom) - p.total_holding_area*10000) < 0.1*st_area(f.geom)
);

-- Each farmplot should have only one crop growing in it
-- delete from
--     filtered_amravati_pahani as fil
-- where
--     fil.key
-- not in (
--     select
--         p.key
--     from
--         public.amravati_pahani as p
--     join
--         s2.filtered_farmplots_amravati as f
--     on
--         st_contains(f.geom, p.geom)
--     where
--         (
--             f.ogc_fid
--         in (
--             select
--                 f.ogc_fid
--             from
--                 s2.filtered_farmplots_amravati as f
--             join
--                 public.amravati_pahani as p
--             on
--                 st_contains(f.geom, p.geom)
--             and
--                 p.sowing_season = 'खरीप'
--             group by
--                 f.ogc_fid
--             having
--                 count(distinct p.crop_name) = 1
--         )
--         or
--             f.ogc_fid
--         not in (
--             select
--                 f.ogc_fid
--             from
--                 s2.filtered_farmplots_amravati as f
--             join
--                 public.amravati_pahani as p
--             on
--                 st_contains(f.geom, p.geom)
--             and
--                 p.sowing_season = 'खरीप'
--             group by
--                 f.ogc_fid
--         )
--     )
--     and
--         (
--             f.ogc_fid
--         in (
--             select
--                 f.ogc_fid
--             from
--                 s2.filtered_farmplots_amravati as f
--             join
--                 public.amravati_pahani as p
--             on
--                 st_contains(f.geom, p.geom)
--             and
--                 p.sowing_season = 'रब्बी'
--             group by
--                 f.ogc_fid
--             having
--                 count(distinct p.crop_name) <= 1
--         )
--         or
--             f.ogc_fid
--         not in (
--             select
--                 f.ogc_fid
--             from
--                 s2.filtered_farmplots_amravati as f
--             join
--                 public.amravati_pahani as p
--             on
--                 st_contains(f.geom, p.geom)
--             and
--                 p.sowing_season = 'रब्बी'
--             group by
--                 f.ogc_fid
--         )
--     )
-- );

select
    count(*)
from
    public.filtered_amravati_pahani;
