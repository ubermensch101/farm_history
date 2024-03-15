drop table if exists
    public.filtered_amravati_pahani;
create table
    public.filtered_amravati_pahani
as
    select
        *
    from
        public.amravati_pahani;

-- Filter 1: Each tuple (khate number, full_name, area) must belong to only one point
delete from
    public.filtered_amravati_pahani
where
    (
        khate_number,
        full_name,
        total_holding_area
    )
in
    (
        select
            khate_number,
            full_name,
            total_holding_area
        from
            public.filtered_amravati_pahani
        group by
            khate_number,
            full_name,
            total_holding_area
        having
            count(distinct geom) > 1
    );

-- Filter 2A: Mishra peek entries must have multiple crops associated with them
delete from
    public.filtered_amravati_pahani
where
    (
        khate_number,
        full_name,
        total_holding_area
    )
in
    (
        select
            khate_number,
            full_name,
            total_holding_area
        from
            public.filtered_amravati_pahani
        where
            sowing_pattern = 'मिश्र पिक' 
        group by
            khate_number,
            full_name,
            total_holding_area,
            sowing_season
        having
            count(crop_name) = 1
    );

-- Filter 2B: Nirbed peek entries must have only one crop associated with them
delete from
    public.filtered_amravati_pahani
where
    (
        khate_number,
        full_name,
        total_holding_area
    )
in
    (
        select
            khate_number,
            full_name,
            total_holding_area
        from
            public.filtered_amravati_pahani
        where
            sowing_pattern = 'निर्भेळ पिक'
        group by
            khate_number,
            full_name,
            total_holding_area,
            sowing_season
        having
            count(crop_name) > 1
    );

-- Filter 3:
