select
	s.name,
	s.telegram_contact
from student s 
where s.city = 'Казань' or s.city = 'Москва'
order by 1 desc;

select
	'университет: ' || c."name" || '; количество студентов: ' || c."size" as """полная информация"""
from college c 
order by 1;

select
	c."name"
	,c."size"
from college c
where c.id  in (10, 30, 50)
order by 2, 1;

select
	c."name"
	,c."size"
from college c
where c.id not in (10, 30, 50)
order by 2, 1;

select
	cr."name"
	,cr.amount_of_students 
from course cr 
where 
	cr.is_online = true
	and cr.amount_of_students between 27 and 310
order by 1 desc, 2 desc;

select 
	cr."name"  
from course cr 
union all
select 
	s."name" 
from student s
order by 1 desc;

select 
	c."name"
	, 'университет' as object_type
from college c  
union all
select 
	cr."name"
	, 'курс' as object_type
from course cr
order by 2 desc, 1 desc;

select 
	c."name"
	,c.amount_of_students 
from course c 
order by 
	case
		when c.amount_of_students = 300 then c.amount_of_students
	end,
	case
		when c.amount_of_students != 300 then c.amount_of_students
	end 
limit 3;

INSERT INTO public.course
(id, "name", is_online, amount_of_students, college_id)
VALUES(60, 'Machine Learning', false, 17,  
	(select c.college_id
	from course c 
	where 
		c."name" = 'Data Mining'));

select
	c.id
from course c 
	except
	select
		id
	from student_on_course soc 
union
select
	id
from student_on_course soc 
except
	select
		c.id
	from course c 
	order by 1;
	
select 
	s."name" as student_name
	,c."name" as course_name
	,cg."name" as student_college
	,soc.student_rating as student_rating 
from student_on_course soc  join student s on (soc.student_id = s.id)
	join course c on (soc.course_id = c.id)
	join college cg on (s.college_id = cg.id and cg."size" > 5000)
where 
	soc.student_rating > 50
order by 1, 2;

select 
	 distinct greatest(s."name", s2."name") as student_1
	 , least(s2."name", s."name") as student_2
	 ,s.city as city
from student s join student s2 on (s.city = s2.city and s."name" != s2."name")
order by 1;

select 
	case 
		when soc.student_rating < 30 then 'неудовлетворительно'
		when soc.student_rating >= 30 and soc.student_rating < 60 then 'удовлетворительно'
		when soc.student_rating >= 60 and soc.student_rating < 85 then 'хорошо'
		when soc.student_rating >= 85 then 'отлично'		
	end as "оценка"
	,count(soc.student_id) as "количество студентов"
from student_on_course soc
group by "оценка"
order by 1

select
	c."name" as "курс"
	,case 
		when soc.student_rating < 30 then 'неудовлетворительно'
		when soc.student_rating >= 30 and soc.student_rating < 60 then 'удовлетворительно'
		when soc.student_rating >= 60 and soc.student_rating < 85 then 'хорошо'
		when soc.student_rating >= 85 then 'отлично'		
	end as "оценка"
	,count(soc.student_id) as "количество студентов"
from student_on_course soc join course c on (soc.course_id = c.id)
group by "оценка", c."name"
order by 1, 2


