

-- Properties and number of graphs computed for
select property, count(graph) as "Graphs Cached" from public.prop_values GROUP BY property;



--Only two very large graphs without a computed 2BG verification
select graph from public.prop_values EXCEPT select graph from public.prop_values where property like 'is_2_bootstrap_good';

--33561 unique graphs with some properties cached
select count(distinct graph) from public.prop_values;

--32988 non-isomorphic graphs that are 2BG
select COUNT(*) from public.prop_values where property like 'is_2_bootstrap_good' and value is TRUE;

--571/33559 Graphs that are not 2BG
select COUNT(*) from public.prop_values where property like 'is_2_bootstrap_good' and value is FALSE;

SELECT property,graph,value FROM prop_values;

--Here be counter examples
select * from public.prop_values where property like 'is_2_bootstrap_good' and value is FALSE;





select * from public.prop_values
    p1 join public.prop_values p2 on p1.graph LIKE p2.graph and p1.property != p2.property
    where p1.property like 'is_2_bootstrap_good' and p1.value is TRUE
    and p2.property like 'is_dirac' and p2.value is TRUE;

CREATE OR REPLACE VIEW pairs as select p1.graph, p1.property as "property_1", p1.value as "value_1", p2.property as "property_2", p2.value as "value_2" from
    public.prop_values p1 join public.prop_values p2 on p1.graph LIKE p2.graph and p1.property != p2.property
    where p1.property like 'is_2_bootstrap_good';

select * from pairs;


select count(distinct graph) from pairs;
select count(distinct graph) from pairs where property_1 like 'is_2_bootstrap_good' and property_2 like 'alpha_leq_order_over_two'
                                          and property_1 NOT LIKE property_2 and value_1 = value_2;


--all graphs with 2bg and another given property computed
select * from (select distinct graph from public.prop_values p1 where p1.property like 'is_2_bootstrap_good'
    and p1.graph in (select distinct graph from public.prop_values where property like 'is_cograph')) t1
    join pairs on t1.graph LIKE pairs.graph and pairs.property_2 like 'is_cograph'
    where value_2 = True and value_1 = False;


select count(*) from public.inv_values where invariant like 'number_of_blocks';


select pairs.graph from (select distinct graph from public.prop_values p1 where p1.property like 'is_2_bootstrap_good'
    and p1.graph in (select distinct graph from public.prop_values where property like 'is_cograph')) t1
    join pairs on t1.graph Like pairs.graph and pairs.property_2 like 'is_cograph'
    where value_2 = True and value_1 = False and pairs.graph in (select distinct graph from public.inv_values inv where inv.invariant like 'number_of_blocks' and inv.value <= 2);



select count(distinct graph) from public.inv_values inv where inv.invariant like 'number_of_blocks' and inv.value <= 2;

select * from public.inv_values;








